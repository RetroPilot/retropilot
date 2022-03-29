# ==============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.utils import io_utils
from abc import abstractmethod, ABCMeta
import shutil
import os
import fnmatch


# ------------------------------------------------------------------------------
#   Custom Op config helpers
# ------------------------------------------------------------------------------

def check_all_equal(iterable):
    return not iterable or iterable.count(iterable[0]) == len(iterable)


def aggregate_property(name, expected_aggregate_type):
    attr_name = '_' + name

    @property
    def prop(self):
        return getattr(self, attr_name)

    @prop.deleter
    def prop(self):
        raise IndexError("Cannot delete this field")

    @prop.setter
    def prop(self, values):

        if not isinstance(values, (list, tuple)):
            raise TypeError('Aggregate value: {} must be a valid object of list or tuple, got : {} instead'
                            .format(name, type(self)))
        for value in values:
            def check_aggregate_type(aggregate_type):
                if not isinstance(value, aggregate_type):
                    raise TypeError('Each aggregate value: {} must be a valid object of type {}'
                                    .format(name, expected_aggregate_type))

            if isinstance(expected_aggregate_type, (list, tuple)):
                map(check_aggregate_type, expected_aggregate_type)
            else:
                check_aggregate_type(expected_aggregate_type)
            if hasattr(self, attr_name):
                raise AttributeError('Cannot set {} field once it has been initialized'.format(attr_name))

        setattr(self, attr_name, values)

    return prop


def union_property(name, expected_union_types):
    attr_name = '_' + name

    @property
    def prop(self):
        return getattr(self, attr_name)

    @prop.deleter
    def prop(self):
        raise IndexError("Cannot delete this field")

    @prop.setter
    def prop(self, value):
        if not isinstance(value, tuple(expected_union_types)):
            raise TypeError('Union value must be a valid object of {}, got : {} instead'
                            .format(expected_union_types, type(self)))
        self.union_type = type(self)
        if hasattr(self, attr_name):
            raise AttributeError('Cannot set {} field once it has been initialized'.format(attr_name))
        setattr(self, attr_name, value)

    return prop


def property_type(name, expected_type):
    attr_name = '_' + name

    @property
    def prop(self):
        return getattr(self, attr_name)

    @prop.deleter
    def prop(self):
        raise IndexError("Cannot delete this field")

    @prop.setter
    def prop(self, value):
        if not isinstance(value, expected_type):
            raise TypeError('{} must be a valid object of type {}'.format(name, expected_type))
        if hasattr(self, attr_name):
            raise AttributeError('Cannot set {} field once it has been initialized'.format(attr_name))
        setattr(self, attr_name, value)

    return prop


class IOHandler:
    """
    Helper class to handle resource management for creating and writing to files/directories.
    """
    __metaclass__ = ABCMeta

    def __init__(self, resource_name, destination, *, copy_source_location=None):
        self.resource_name = resource_name
        self.destination = destination
        self.copy_source_location = copy_source_location
        self.rendered = False
        self.writable_content = None
        self.isopen = False
        self.reversible = True

    def __enter__(self):
        self.isopen = True
        return self

    @abstractmethod
    def render(self, *, force_generation=False, writable_content=None, **io_args):
        """
         Abstract class that controls how the handler creates the underlying resource.
       """

    @abstractmethod
    def revert(self):
        """
        Reverts a resource to its previous state if it has been rendered
        :return:
        """

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_tb:
            pass
        else:
            print(exc_val, exc_type)
            print(exc_tb)

    def check_is_open(self):
        if not self.isopen:
            raise BlockingIOError("Cannot render file unless explicitly opened")

    def set_writable_content(self, writable_content):
        if not isinstance(writable_content, dict):
            raise TypeError(
                "Writable content for handler must be a dictionary of {resource_name: content}")
        self.writable_content = writable_content


class FileHandler(IOHandler):
    """
    This manages how a file is created, and how content is written to it if any.
    """

    def __init__(self, file_name, file_destination, *, copy_source_location=None):
        super(FileHandler, self).__init__(file_name, file_destination, copy_source_location=copy_source_location)
        self.file_handle = None
        self.file_abs_path = os.path.abspath(os.path.join(os.path.abspath(file_destination), file_name))

    def render(self, *, force_generation=True, **io_args):
        """
        This function creates a file and writes content to it if its writable content member is not empty.
        :param force_generation: If set to false, the function attempts to append content rather than overwriting and
                                 vice versa. If the file is to be copied, it is skipped.
        :param io_args: explicit IO module args to use for writing any writable content
        :return:
        """
        self.check_is_open()
        if os.path.exists(self.file_abs_path) and not force_generation:
            log_warning('File: {} exists. File mode should be set to append to avoid overwriting file. '
                        'Note that if a file is to be copied, it will be skipped.'
                        .format(self.resource_name))

        if not self.copy_source_location:
            if not self.file_handle:
                log_debug("Creating new file at: {}".format(self.file_abs_path))
                if io_args:
                    self.file_handle = open(self.file_abs_path, **io_args)
                elif force_generation:
                    self.reversible = False
                    self.file_handle = open(self.file_abs_path, 'wt+')
                else:
                    self.file_handle = open(self.file_abs_path, 'w')
            if self.writable_content is not None and self.resource_name in self.writable_content:
                self.file_handle.write(str(self.writable_content.get(self.resource_name)))
            log_debug("File:{} generation complete to: {}", self.resource_name, self.destination)
        else:
            io_utils.check_validity(self.copy_source_location)
            if force_generation or not os.path.exists(self.file_abs_path):
                shutil.copy2(self.copy_source_location, self.file_abs_path)  # this will overwrite files
                log_debug("File:{} copied to: {}", self.resource_name, self.destination)
            else:
                self.reversible = False  # dont do anything if force generation is not set
                pass

    def revert(self):
        """
        Reverts a file that was created by deleting it, provided it can be reverted. If the file is open, then it is
        only if it not rendered and can be safely reverted.
        :return:
        """
        self.check_is_open()
        if not self.rendered:
            return
        elif self.reversible:
            os.remove(self.file_abs_path)
            log_info("Removing file: {} from location: {}".format(self.resource_name, self.destination))

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_tb:
            if self.file_handle:
                self.file_handle.close()
            self.rendered = True
        else:
            print(exc_val, exc_type)
            print(exc_tb)


class DirectoryHandler(IOHandler):
    """
    The purpose of this class is to create directory structures in memory, such that a directory handler can control
    the creation of its subdirectories and writing of its constituent files. A single directory handler can nest several
    directory handlers, which can further nest file handlers in a recursive fashion. E.x

    Dir A contains Dir B, Dir C, File A
    Dir B contains File B, File C
    Dir C contains Dir D, File E

    Dir A.render() -> Dir B.render(), Dir C.render(), File B.render ..... and so on.

    On a single render call, each nested element is created and tracked. Upon failure, all parents are reverted from the
    current element to the root. For files, writable content can be rendered using the same render member function,
    provided the content is set using the set_writable_content method.

    """

    def __init__(self, dir_name, dir_destination, *, copy_source_location=None, file_handlers=None, dir_handlers=None):
        """
        Initializes a directory handler
        :param dir_name: The name of the directory
        :param dir_destination: The destination it will be created (note this is the base name)
        :param copy_source_location: The location it will be copied from, if it should not be created afresh
        :param file_handlers: A list of file handlers to associate with this directory
        :param dir_handlers: A list of directory handlers to associate with this directory
        """
        super(DirectoryHandler, self).__init__(dir_name, dir_destination, copy_source_location=copy_source_location)
        self.dir_abs_path = os.path.abspath(os.path.join(os.path.abspath(dir_destination), dir_name))
        if file_handlers:
            self.file_handlers = file_handlers
        else:
            self.file_handlers = []
        if dir_handlers:
            self.dir_handlers = dir_handlers  # should check dir handlers type
        else:
            self.dir_handlers = []

    def render(self, *, force_generation=True, **io_args):
        """
        Renders a directory handler and all its children which may be file handlers or directory handlers nested
        recursively. The resource is marked rendered by each successively handler if there are no Exceptions.

        :param force_generation: If force generation is set, then a new directory is always created. Otherwise,
                                 creation is skipped, and the handler is marked as reversible
        :param io_args: Additional io args to specify if the directory handler contains file handlers
        """
        self.check_is_open()
        if not io_args:
            io_args = {'mode': 'wt+'}
        if os.path.exists(self.dir_abs_path):
            if not force_generation:
                self.reversible = False
                log_warning('Directory: {} exists. '
                            'File mode should be set to append to avoid overwriting files in directory'
                            .format(self.resource_name))
            else:
                log_warning("Force generation is set. Deleting existing directory at {}".format(self.dir_abs_path))
                shutil.rmtree(self.dir_abs_path)
                log_debug("Creating new directory at: {}".format(self.dir_abs_path))
                os.makedirs(self.dir_abs_path)
        else:
            log_debug("Creating new directory at: {}".format(self.dir_abs_path))
            os.makedirs(self.dir_abs_path)

        if not self.copy_source_location:
            for handler in self.dir_handlers + self.file_handlers:
                with handler as h:
                    h.set_writable_content(self.writable_content)
                    h.render(force_generation=force_generation, **io_args)
        else:
            io_utils.check_validity(self.copy_source_location, is_directory=True)
            self.copy_dir(self.resource_name, self.copy_source_location, self.destination)

    @staticmethod
    def copy_dir(dir_to_copy, origin, copy_location, unwanted_patterns=None):
        """
        Copies a directory from an origin to destination, excluding any files that match a certain pattern.

        :param dir_to_copy: The directory name to be copied
        :param origin: The origin of the directory
        :param copy_location: The location it should be copied to
        :param unwanted_patterns: Any unwanted patterns to exclude such as *.pyc or *CmakeList.txt etc.
        :return:
        """
        if unwanted_patterns is None:
            unwanted_patterns = list()
        if dir_to_copy:
            dir_location = os.path.join(copy_location, dir_to_copy)
            origin_dir_location = os.path.join(origin, dir_to_copy)
            if not os.path.exists(dir_location):
                log_debug_msg_as_status("Creating directory and copying files")
                shutil.copytree(src=origin_dir_location,
                                dst=dir_location, ignore=shutil.ignore_patterns(*unwanted_patterns))
            else:
                log_debug("Directory {} exists! Attempting to copy distinct files into existing directory.",
                          copy_location)
                for elem in os.listdir(origin_dir_location):
                    if os.path.isdir(os.path.join(origin_dir_location, elem)):
                        dir_ = elem
                        DirectoryHandler.copy_dir(dir_,
                                                  origin_dir_location,
                                                  dir_location,
                                                  unwanted_patterns)

                    else:
                        file = elem
                        if file not in os.listdir(dir_location) and not ([fnmatch.filter(file, pattern) for pattern
                                                                          in unwanted_patterns]):
                            shutil.copy2(src=os.path.join(origin_dir_location, file),
                                         dst=os.path.join(dir_location, file))
                        else:
                            log_debug("File exists! Skipping {}", file)
                        log_debug("Files copied from {} to {}", origin, copy_location)

    def get_handler(self, resource_name):
        """
        Retrieves a handler if it is a child of the current parent instance
        :param resource_name: The name of the resource
        """
        for handler in self.file_handlers:
            if handler.resource_name == resource_name:
                return handler
        for handler in self.dir_handlers:
            if handler.resource_name == resource_name:
                return handler
            else:
                child_handler = handler.get_handler(resource_name)
                if child_handler != None:
                    return child_handler
        return None

    def revert(self, resource_name=None):
        """
        Reverts the current resource provided it exists, and any additional resource specified as a resource name.
        The resource can only be reverted if it has been rendered, and is marked as reversible.
        :param resource_name:
        :return:
        """
        self.check_is_open()
        if resource_name:
            retrieved_resource = self.get_handler(resource_name)
            # nothing to do if resource does not exist or has been explicitly marked as non-reversible
            if retrieved_resource is not None and retrieved_resource.reversible:
                retrieved_resource.revert()
        if os.path.exists(self.dir_abs_path) and self.reversible:
            shutil.rmtree(self.dir_abs_path)
            log_info("Removing directory: {} from location: {}".format(self.resource_name, self.destination))

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_tb:
            for handler in self.file_handlers + self.dir_handlers:
                if not handler.rendered:
                    handler.revert()
                    raise IOError("Could not create:{} in directory: {}".format(handler.resource_name,
                                                                                self.resource_name))
            self.rendered = True
            if not self.copy_source_location:
                log_debug("Directory:{} generation complete to: {}", self.resource_name, self.destination)
            else:
                log_debug("File:{} copied to: {}", self.resource_name, self.destination)
        else:
            for handler in self.file_handlers + self.dir_handlers:
                self.revert(handler.resource_name)
            self.revert()
            print(exc_val, exc_type)
            print(exc_tb)


class CustomOpNotFoundError(Exception):
    pass
