# ==============================================================================
#
#  Copyright (c) 2020 - 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from .core import *
from qti.aisw.converters.common.utils.converter_utils import *
import json
import os
import shutil
from itertools import chain
import fnmatch

dsp_detected = False


# ------------------------------------------------------------------------------
#   Udo config Enum Style Classes
# ------------------------------------------------------------------------------
class UdoTemplateFileReader:
    """ Enum class that stores template file names and their corresponding types"""

    template_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'templates')

    # Note templates are repeated with the assumption that they may need to be different across runtimes.
    # GPU and CPU share implementations
    # DSP Udo Implementations are yet to be finalized.
    # The other templates are explained by the naming convention. New templates should be added here and placed into
    # corresponding file type.

    TEMPLATE_FILES = [['reg_lib_template.mako'],

                      ['impl_lib_cpu_template.mako',
                       'op_impl_lib_cpu_gpu_header.mako',
                       'op_impl_lib_cpu_gpu_source.mako'],

                      ['impl_lib_dsp_template.mako',
                       'op_impl_lib_dsp_header.mako',
                       'op_impl_lib_dsp_source.mako',
                       'impl_lib_htp_template.mako',
                       'op_impl_lib_htp_source.mako'],

                      ['impl_validation_header.mako',
                       'impl_validation_header_source.mako'],

                      ['reg_makefile_template.mako',
                       'android_makefile_template.mako',
                       'runtime_makefile_template.mako',
                       'main_makefile_template.mako'],

                       ['root_cmakelists_template.mako',
                        'lib_cmakelists_template.mako'],
                      ]
    TEMPLATE_FILE_TYPES = {'regfile': 0,
                           'cpuimplfile': 1,
                           'gpuimplfile': 1,
                           'dspimplfile': 2,
                           'validationimplfile': 3,
                           'makefile': 4,
                           'cmakefile': 5}


class UdoPackageStatus(object):
    """
    This class contains the possible statuses of a udo package
    """
    NOT_GENERATED = 0
    GENERATED_NOT_IMPLEMENTED = 1
    IMPLEMENTED = 2
    PACKAGE_CAN_COMPILE = 3  # for sanity checking, may not be exposed in production


# ------------------------------------------------------------------------------
#   Udo config Core Classes
# ------------------------------------------------------------------------------
class UdoGenerator:
    """
    This class is the main point of entry for the package-generator. It handles the parsing of the user provided
    config and creates a udo package object. It then sets up file paths according to information gathered in the
    package in the parsing step. Finally, implements the package by auto-generating header and source files using a
    UdoFileGenerator object.

    :udo_packages: This is a list of all udo package registered in a single generator instance
    :UdoFileGenerator: This object handles the auto-generation of code using Mako templates.
    """

    def __init__(self):
        self.udo_packages = list()
        self.UdoFileGenerator = UdoCodeGenerator

    def register_package(self, udo_package):
        if udo_package not in self.udo_packages:
            self.udo_packages.append(udo_package)

    def parse_config(self, config_path, output_path=None):
        """
        Parses a user provided json config into a udo package object. The config is expected to contain information
        about a user's operation, as well as additional fields about the package.
        :param config_path: The file path to the user's json config file
        :param output_path: The output path for where the package will be save. It will override UDO_PACKAGE_PATH if
        provided
        """
        # Import config
        with open(config_path, 'r') as json_config:
            config_vars = json.load(json_config)

        log_debug_msg_as_status("Parsing config: {}", config_path)
        for udo_package_name, udo_package_dict in config_vars.items():
            new_udo_package = UdoPackage(udo_package_dict['UDO_PACKAGE_NAME'])
            udo_package_info = UdoPackageInfo.from_dict(udo_package_dict)
            if output_path:
                if udo_package_info.root:
                    log_debug("Output path also defined in config. "
                              "Overriding with output path defined on command line: {}",
                              output_path)
                udo_package_info.root = os.path.join(os.path.realpath(output_path),
                                                     udo_package_info.name)
            new_udo_package.add_package_info(udo_package_info)
            self.register_package(new_udo_package)
        log_debug("Config parsed.")

    def setup_file_paths(self, ignore_includes=True, force_generation=False, **files):
        """
         This sets up file paths and makes the directory structure for the package. It makes the top level directory,
         followed by the src, lib, include and makefile directories. This method will not overwrite directories if they
         already exist.
        :param force_generation:  if set to true, any package directory in the generator instance will be overwritten
                                  if it exists.
        :param ignore_includes: setting this flag to false means files will be copied from SnpeUdo API into
                              the package. If this flag is set to ignore, the user gets a warning as the files will be needed
                              during compilation
        :param files: These are files the user may want to copy into their created package. The current user is to copy
                      the config into this directory.
        :return: the package paths that have been successfully set up
        """
        # for udo package in udo_packages
        # setup root-> lib, src, include, *.json
        udo_package_paths = []
        unwanted_patterns = []
        global dsp_detected

        for udo_package in self.udo_packages:

            SNPE_UDO_ROOT = udo_package.package_info.SNPE_UDO_ROOT
            udo_root = udo_package.root
            if not SNPE_UDO_ROOT:
                raise IOError("Files cannot be copied as "
                              "SNPE_UDO_ROOT variable is not set.")

            if udo_package.status > UdoPackageStatus.NOT_GENERATED:
                log_warning('Udo package files have already been generated. Possible duplicate')

            if os.path.exists(udo_root):
                if not force_generation:
                    log_warning("{} already exists! File generation may be incomplete."
                                " Please specify -f to force complete generation on an existing directory."
                                .format(udo_root))
                else:
                    if not os.path.isdir(udo_root):
                        raise TypeError("{} already exists but it is not a directory. "
                                        "Please specify a unique directory name".format(udo_root))

                    log_warning(
                        "Force generation is set. Deleting existing udo Package at {}".format(
                            udo_root))
                    shutil.rmtree(udo_root)
                    os.makedirs(udo_root)
            else:
                log_info("Creating new package at: {}".format(udo_root))
                os.makedirs(udo_root)

            # Make runtime specific checks and set warnings as needed
            gpu_detected = False
            if "GPU" in udo_package.supported_runtimes:
                if not os.getenv("CL_INCLUDE_PATH", None):
                    log_warning('GPU Operation detected but CL_INCLUDE_PATH is not set. Please note'
                                ' CL_INCLUDE_PATH needs to be set to compile the package.')
                if not os.getenv("CL_LIBRARY_PATH", None):
                    log_warning('GPU Operation detected but CL_LIBRARY_PATH is not set. Please note'
                                ' CL_LIBRARY_PATH needs to be set to compile the package.')
                gpu_detected = True

            if 'DSP' in udo_package.supported_runtimes:
                if not os.getenv("HEXAGON_SDK_ROOT", None):
                    log_warning(
                        'DSP Operation detected but HEXAGON_SDK_ROOT is not set. Please note'
                        ' HEXAGON_SDK_ROOT needs to be set to compile the package.')
                dsp_detected = True

            # Initialize directories
            log_debug_msg_as_status("Creating directories in package")
            jni_dir = os.path.join(udo_root, 'jni')
            src_dir = os.path.join(jni_dir, 'src')
            util_src_dir = os.path.join(src_dir, 'utils')
            include_dir = os.path.join(udo_root, 'include')
            config_dir = os.path.join(udo_root, 'config')
            makefiles_dir = os.path.join(udo_root, 'Makefiles')

            # create list for unwanted patterns
            unwanted_patterns = []

            # make jni dir to mimic android dir structure
            self.make_udo_dir(jni_dir)

            # make runtime specific directories for src file which will be placed in jni
            self.make_udo_dir(src_dir, per_runtime=True, runtimes=udo_package.supported_runtimes,
                              dsp_arch_types=udo_package.dsp_arch_types)

            # make registration directory for source files which will be placed in jni
            self.make_udo_dir(os.path.join(src_dir, 'reg'))

            # make include and config directories
            self.make_udo_dir(include_dir)
            self.make_udo_dir(config_dir)

            # copy Makefiles, util files, header files into newly created package
            log_debug_msg_as_status("Copying files from SDK")
            if ignore_includes:
                log_debug_msg_as_status('Skipping SnpeUdo include files')
            else:
                # copy SnpeUdo includes, Note SNPE_ROOT must be set correctly
                SNPE_ROOT = os.getenv("SNPE_ROOT")
                if not SNPE_ROOT:
                    raise RuntimeError("SNPE_ROOT needs to be set to use this option")
                self.copy_udo_dir("SnpeUdo", os.path.join(SNPE_ROOT, "include", "zdl"),
                                  os.path.join(include_dir, "SnpeUdo"), unwanted_patterns + [
                                      '*.cpp', '*.c'])

            # Temporary fix until directory structure is confirmed
            # set unwanted patterns to remove OpenCl dependency for Cpu
            if not gpu_detected:
                unwanted_patterns = ['*GPU*']

            # Copy util include files
            self.copy_udo_dir("utils", SNPE_UDO_ROOT,
                              os.path.join(include_dir, "utils"), unwanted_patterns + [
                                  '*.cpp', '*.c'])

            # copy util src files
            self.copy_udo_dir('utils', SNPE_UDO_ROOT, util_src_dir,
                              unwanted_patterns + ['*.h', '*.hpp'])

            # copy application make file
            android_makefile = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            'makefiles',
                                            'Application.mk')
            if not os.path.exists(os.path.join(jni_dir, 'Application.mk')):
                shutil.copy2(android_makefile, os.path.join(jni_dir, 'Application.mk'))

            # copy Makefiles except application.mk
            common_makefile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'makefiles',
                                           'common.mk')
            shutil.copy2(common_makefile, os.path.join(udo_root, 'common.mk'))
            # copy other platforms' Makefile
            for compiler in ['aarch64-linux-gcc4.9', 'aarch64-qnx-gcc5.4']:
                makefile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'makefiles',
                                        'Makefile.%s' % compiler)
                if os.path.exists(makefile):
                    shutil.copy2(makefile, os.path.join(udo_root, os.path.basename(makefile)))

            # copy user files into package
            if files:
                for file_path in files.values():
                    if file_path:
                        log_debug_msg_as_status("Copying config file")
                        if not os.path.exists(
                                os.path.join(config_dir, os.path.basename(file_path))):
                            shutil.copy(src=file_path, dst=config_dir)
                            log_debug("Config file copied from {} to {}", file_path, config_dir)
                        else:
                            log_debug("File exists! Skipping {}", file_path)

            # make sure the udo package keeps track of where it is in the process
            log_debug("Package directory setup and requested files have been copied.")
            udo_package.status = UdoPackageStatus.GENERATED_NOT_IMPLEMENTED
            udo_package_paths.append([udo_package.root])

        return udo_package_paths

    def implement_packages(self, gen_cmake=False):
        """
         This class handles the implementation of the each provided package by following the following stages:
        - registration files are auto-generated
        - validation files are auto-generated
        - makefiles are generated
        - implementation files are auto-generated
        """
        # Registration, Validation and Makefile Generation
        log_debug_msg_as_status("Auto-generating package code")
        for udo_package in self.udo_packages:
            # Set up registration file paths then generate templates
            jni_package_root = os.path.join(udo_package.root, 'jni')
            reg_file_path = os.path.join(jni_package_root, 'src', 'reg',
                                         udo_package.name + 'RegLib.cpp')
            reg_file_generator = self.UdoFileGenerator(reg_file_path, 'regfile')

            log_debug_msg_as_status("Generating registration files")
            reg_file_generator.substitute_templates(udo_package)
            log_debug("Registration file generation complete")

            # Set up validation paths then generate templates
            for runtime in udo_package.supported_runtimes:
                validate_file_paths = list()
                validate_file_paths.append(
                    os.path.join(udo_package.root, 'include', udo_package.name + runtime.title()
                                 + 'ImplValidationFunctions.hpp'))
                validate_file_paths.append(os.path.join(jni_package_root, 'src', 'reg',
                                                        udo_package.name + runtime.title() + 'ImplValidationFunctions.cpp'))
                validate_file_generator = self.UdoFileGenerator(validate_file_paths,
                                                                'validationimplfile')

                log_debug_msg_as_status(
                    "Generating validation files for {} runtime".format(runtime))
                validate_file_generator.substitute_templates(udo_package, runtime=runtime)
                log_debug("Validation file generation complete")

            # Set up makefile paths for android and registration lib then generate templates
            makefile_file_paths = [(os.path.join(jni_package_root,
                                                 'Android.mk')),
                                   os.path.join(jni_package_root, 'src', 'reg',
                                                'Makefile'),
                                   os.path.join(udo_package.root, 'Makefile')]
            # set up makefile paths per runtime
            for runtime in udo_package.supported_runtimes:
                makefile_file_paths.append(os.path.join(jni_package_root, 'src', runtime.upper(),
                                                        'Makefile'))

            make_file_generator = self.UdoFileGenerator(makefile_file_paths, 'makefile')

            log_debug_msg_as_status("Generating make files")
            make_file_generator.substitute_templates(udo_package,
                                                     runtimes=udo_package.supported_runtimes,
                                                     dsp_arch_types=udo_package.dsp_arch_types)
            log_debug("Makefile generation complete")

            # Implementation File Generation
            for operator in udo_package.package_info.operators:
                op_name = operator.type_name
                op_runtimes = list(
                    map(title_case,
                        [SnpeUdoConstants.snpe_udo_coretypes[x] for x in operator.core_types]))
                for i, runtime in enumerate(op_runtimes):
                    if runtime.lower() != 'dsp':
                        impl_file_paths = [
                            os.path.join(jni_package_root, 'src', str(runtime).upper(),
                                         '{}ImplLib{}.cpp'.format(udo_package.name, str(runtime))),
                            os.path.join(udo_package.root, 'include',
                                         '{}ImplLib{}.hpp'.format(op_name, str(runtime))),
                            os.path.join(jni_package_root, 'src', str(runtime).upper(),
                                         '{}ImplLib{}.cpp'.format(op_name, str(runtime)))]
                    else:
                        impl_file_paths = [
                            os.path.join(jni_package_root, 'src', str(runtime).upper(),
                                         '{}ImplLib{}.c'.format(udo_package.name, str(runtime))),
                            os.path.join(udo_package.root, 'include',
                                         '{}ImplLib{}.h'.format(op_name, str(runtime))),
                            os.path.join(jni_package_root, 'src', str(runtime).upper(),
                                         '{}ImplLib{}.c'.format(op_name, str(runtime))),
                            os.path.join(jni_package_root, 'src', str(runtime).upper(),
                                         '{}ImplLib{}.cpp'.format(udo_package.name, str(runtime))),
                            os.path.join(jni_package_root, 'src', str(runtime).upper(),
                                         '{}ImplLib{}.cpp'.format(op_name, str(runtime)))]
                    impl_file_generator = self.UdoFileGenerator(impl_file_paths,
                                                                '{}implfile'.format(
                                                                    str(runtime).lower()))
                    log_debug_msg_as_status(
                        "Generating {} implementation files for {} operation".format(runtime,
                                                                                     operator.type_name))
                    impl_file_generator.substitute_templates(udo_package, operator=operator,
                                                             core_type=operator.core_types[i],
                                                             runtime=runtime,
                                                             dsp_arch_types=operator.dsp_arch_types)
                    log_debug("Implementation file generation complete")

            # CMakeLists.txt Generation
            if gen_cmake:
                # log warning for those unsupported CMakeLists.txt
                src_package_root = os.path.join(jni_package_root, 'src')
                with os.scandir(src_package_root) as it:
                    for entry in it:
                        if entry.is_dir() and entry.name.lower() not in ['reg', 'utils', 'cpu']:
                            rel_src_package_root = os.path.relpath(src_package_root, os.path.dirname(udo_package.root))
                            log_warning(
                                'CMakeLists for {} is not supported!'
                                .format(os.path.join(rel_src_package_root, entry.name)))
                self.__implement_cmake_package(udo_package)

            log_debug("Code generation is complete for package: {}", udo_package.name)
            udo_package.status = UdoPackageStatus.IMPLEMENTED

    def __implement_cmake_package(self, udo_package):
        # Preparing output file paths
        src_root = os.path.join(udo_package.root, 'jni', 'src')
        output_file_paths = [os.path.join(udo_package.root, 'CMakeLists.txt'),
                             os.path.join(src_root, 'reg', 'CMakeLists.txt')]
        for runtime in udo_package.supported_runtimes:
            if str(runtime).upper() == 'DSP':
               output_file_paths.extend([
                   os.path.join(src_root, 'DSP_' + arch.upper(), 'CMakeLists.txt') for arch in udo_package.dsp_arch_types])
            else:
               output_file_paths.append(os.path.join(src_root, str(runtime).upper(), 'CMakeLists.txt'))

        # TODO: Here we are globbing the src file names assuming they are created correctly before.
        #       It would be better to get those file names and check them explictly here.
        path_to_src_files_dict = dict()
        for file_path in output_file_paths:
            dirname = os.path.dirname(file_path)
            if os.path.isdir(dirname):
                path_to_src_files_dict[file_path] = [ f for f in os.listdir(dirname) if f.endswith(".cpp") ]
            else:
                log_error("Directory does not exist: {}".format(dirname))
                patch_to_src_files_dict[file_path] = []
        util_src_dir = os.path.join(src_root, 'utils')
        util_src_files = [ f for f in os.listdir(util_src_dir) if f.endswith(".cpp") ]

        cmake_file_generator = self.UdoFileGenerator(output_file_paths, 'cmakefile')
        log_debug_msg_as_status("Generating CMakeLists.txt files")
        cmake_file_generator.substitute_templates(udo_package,
                                                  util_src_files=util_src_files,
                                                  path_to_src_files_dict=path_to_src_files_dict)
        log_debug("CMakeLists.txt generation complete")

    def generation_is_complete(self):
        """
        Performs a final check of the package status to ensure it is in the right stage. if the package status is not
        IMPLEMENTED then a debug message will be returned, in addition to boolean false.
        :return: returns True if the package can compile otherwise return False
        """
        for udo_package in self.udo_packages:
            if udo_package.status == UdoPackageStatus.GENERATED_NOT_IMPLEMENTED:
                log_debug(
                    "Package files for {} have been generated but code could not be auto-implemented",
                    udo_package.name)
                return False
            elif udo_package.status == UdoPackageStatus.NOT_GENERATED:
                log_debug("Package files have not been created for package: ", udo_package.name)
                return False
            log_debug("All packages files have been created at: {}", udo_package.root)
            udo_package.status = UdoPackageStatus.PACKAGE_CAN_COMPILE

        return True

    @staticmethod
    def make_udo_dir(dir_path, per_runtime=False, runtimes=None, dsp_arch_types=[]):
        def setup_runtime_dirs():
            for runtime in runtimes:
                if runtime == 'DSP' and dsp_arch_types:
                    for arch in dsp_arch_types:
                        runtime_dir_path = os.path.join(dir_path, 'DSP_' + arch.upper())
                        log_debug_msg_as_status("Creating runtime directory: {}", runtime_dir_path)
                        if not os.path.exists(runtime_dir_path):
                            os.makedirs(runtime_dir_path)
                            log_debug(" {} runtime directory created", runtime)
                        else:
                            log_debug("Directory Exists! Skipping directory: {}", runtime_dir_path)
                else:
                    runtime_dir_path = os.path.join(dir_path, runtime)
                    log_debug_msg_as_status("Creating runtime directory: {}", runtime_dir_path)
                    if not os.path.exists(runtime_dir_path):
                        os.makedirs(runtime_dir_path)
                        log_debug(" {} runtime directory created", runtime)
                    else:
                        log_debug("Directory Exists! Skipping directory: {}", runtime_dir_path)

        if not os.path.exists(dir_path):
            log_debug_msg_as_status("Creating directory: {}", dir_path)
            os.makedirs(dir_path)
            if not per_runtime:
                log_debug("Directory created")
                return
            else:
                setup_runtime_dirs()
        elif per_runtime:
            setup_runtime_dirs()
        else:
            log_debug("Directory Exists! Skipping directory: {}", dir_path)

    @staticmethod
    def copy_udo_dir(dir_to_copy, origin, copy_location, unwanted_patterns=list()):
        unwanted_patterns.extend(['*.txt', '*.orig', '*.pyc'])
        if dir_to_copy:
            if not os.path.exists(copy_location):
                log_debug_msg_as_status("Creating directory and copying files")
                shutil.copytree(src=os.path.join(origin, dir_to_copy),
                                dst=copy_location,
                                ignore=shutil.ignore_patterns(*unwanted_patterns))
            else:
                log_debug(
                    "Directory {} exists! Attempting to copy distinct files into existing directory.",
                    copy_location)
                for file in os.listdir(os.path.join(origin, dir_to_copy)):
                    if file not in os.listdir(copy_location) and not (
                    [fnmatch.filter(file, pattern) for pattern
                     in unwanted_patterns]):
                        shutil.copy2(src=os.path.join(origin, dir_to_copy, file),
                                     dst=os.path.join(copy_location, str(file)))
                    else:
                        log_debug("File exists! Skipping {}", file)
            log_debug("Files copied from {} to {}", origin, copy_location)


class UdoCodeGenerator:
    """
    Handles the code generation of files by performing template substitution using user provided information.
    """

    def __init__(self, output_file_path, file_type):
        self.reader = UdoTemplateFileReader
        self.output_file_path = output_file_path
        self.file_type = self.reader.TEMPLATE_FILE_TYPES[file_type]
        self._template_files = self.get_templates(self.file_type)

    def substitute_templates(self, udo_package, **args):
        """
        Substitute templates strings in a template file with user provided information.
        :param udo_package: the udo package which contain the files that will be auto-generated
        :param args: optional fields that are specific to the template that will be implemented
        """
        if self.file_type == self.reader.TEMPLATE_FILE_TYPES['regfile']:
            self.__substitute_regfile_templates(udo_package)
        elif self.file_type == self.reader.TEMPLATE_FILE_TYPES['validationimplfile']:
            self.__substitute_validatefile_templates(udo_package, args['runtime'])
        elif self.file_type == self.reader.TEMPLATE_FILE_TYPES['makefile']:
            self.__substitute_makefile_templates(udo_package, args['runtimes'],
                                                 args['dsp_arch_types'])
        elif self.file_type == self.reader.TEMPLATE_FILE_TYPES['cmakefile']:
            self.__substitute_cmakefile_templates(udo_package, args['util_src_files'], args['path_to_src_files_dict'])
        else:
            self.__substitute_implfile_templates(udo_package, args['operator'], args['core_type'],
                                                 args['runtime'],
                                                 args['dsp_arch_types'])

    def __substitute_regfile_templates(self, udo_package):
        self.render_templates(self._template_files[0],
                              self.output_file_path,
                              multiple=False,
                              package=udo_package)

    def __substitute_implfile_templates(self, udo_package, operator, core_type, runtime,
                                        dsp_arch_types):
        if runtime.lower() == 'dsp' and dsp_arch_types:
            for i, arch in enumerate(dsp_arch_types):
                self.render_templates(
                    self._template_files[-2:] if int(str(arch)[-2:]) >= 68 else self._template_files[:3],
                    [('DSP_' + arch.upper()).join(path.rsplit('DSP', 1))
                     for path in
                     (self.output_file_path[-2:] if int(str(arch)[-2:]) >= 68 else self.output_file_path[:3])],
                    package=udo_package,
                    package_name=udo_package.name,
                    op_name=operator.type_name,
                    operator=operator,
                    dsp_arch_type=arch.lower(),
                    core_type=core_type,
                    runtime=runtime)
        else:
            self.render_templates(self._template_files[:3],
                                  self.output_file_path[:3],
                                  package=udo_package,
                                  package_name=udo_package.name,
                                  op_name=operator.type_name,
                                  operator=operator,
                                  dsp_arch_type='v60',
                                  core_type=core_type,
                                  runtime=runtime)

    def __substitute_validatefile_templates(self, udo_package, runtime):
        operators = [operator for operator in udo_package.package_info.operators for core_type in
                     operator.core_types
                     if SnpeUdoConstants.snpe_udo_coretypes[core_type].lower() == runtime.lower()]
        self.render_templates(self._template_files,
                              self.output_file_path,
                              package_name=udo_package.name,
                              runtime=runtime.title(),
                              operators=operators)

    def __substitute_makefile_templates(self, udo_package, runtimes, dsp_arch_types):

        # android makefile
        self.render_templates(self._template_files[1],
                              self.output_file_path.pop(0),
                              multiple=False,
                              package=udo_package)

        # registration makefile
        self.render_templates(self._template_files[0],
                              self.output_file_path.pop(0),
                              multiple=False,
                              package=udo_package)

        # main makefile
        self.render_templates(self._template_files[3],
                              self.output_file_path.pop(0),
                              multiple=False,
                              dsp_arch_types=dsp_arch_types,
                              package=udo_package)

        # runtime specific makefiles, intended use is for android/embedded-linux targets.
        for i, runtime in enumerate(runtimes):
            if runtime.lower() == 'dsp' and dsp_arch_types:
                for j, arch in enumerate(dsp_arch_types):
                    self.render_templates(self._template_files[2],
                                          ('DSP_' + arch.upper()).join(
                                              self.output_file_path[i].rsplit('DSP', 1)),
                                          multiple=False,
                                          package=udo_package,
                                          dsp_arch_type=arch.lower(),
                                          runtime=title_case(runtime))
            else:
                self.render_templates(self._template_files[2],
                                      self.output_file_path[i],
                                      multiple=False,
                                      package=udo_package,
                                      dsp_arch_type='v60',
                                      runtime=title_case(runtime))

    def __substitute_cmakefile_templates(self, udo_package, util_src_files, path_to_src_files_dict):
        subdirs = []
        # reg/runtime CMakeLists.txt
        for file_path in self.output_file_path[1:]:
            dirname = os.path.basename(os.path.dirname(file_path))
            lib_name_suffix = ''
            src_files = path_to_src_files_dict[file_path]
            runtimes_of_lib_def = []
            if dirname.lower() == 'reg':
                lib_name_suffix = 'Reg'
                runtimes_of_lib_def = [str(runtime) for runtime in udo_package.supported_runtimes if str(runtime).lower() != 'dsp']
            elif dirname.lower() == 'cpu':
                lib_name_suffix='Impl' + dirname.title()
            else:
                continue
            self.render_templates(self._template_files[1],
                                  file_path,
                                  multiple=False,
                                  package_name=udo_package.name,
                                  lib_name_suffix=lib_name_suffix,
                                  util_src_files=util_src_files,
                                  src_files=src_files,
                                  runtimes_of_lib_def=runtimes_of_lib_def)
            subdirs.append(os.path.dirname(os.path.relpath(file_path, udo_package.root)))
        # root CMakeLists.txt
        self.render_templates(self._template_files[0],
                              self.output_file_path[0],
                              multiple=False,
                              package_name=udo_package.name,
                              subdirs=subdirs)

    def get_templates(self, file_type):
        return list(map(lambda x: os.path.join(self.reader.template_path, x),
                        self.reader.TEMPLATE_FILES[file_type]))

    def render_templates(self, template_files, out_files, multiple=True, **args):
        """
         This method handles the template substitution by calling mako on templates that have been created for the
         package. Mako subtitutes the template fields with the user provided information and
         returns a rendered template. The rendered template is saved in output files provided.

        :param template_files:The list of template files to be substituted
        :param out_files: The file locations where the rendered template will be saved/
        :param multiple:  Indicates that multiple templates are to be used, and there will be more than one output file.
                          Default value is true.
        :param args: Optional arguments that are specific to the set of templates provided.
        """
        try:
            import mako
            from mako.lookup import TemplateLookup
            from mako.template import Template
        except ImportError as e:
            raise Exception(
                "{}: Mako template dependency not found. Please ensure mako is installed".format(
                    type(e)))

        mytemplate = ''
        if not multiple:
            template_files = [template_files]
            out_files = [out_files]

        template_dir = self.reader.template_path
        directory_lookup = TemplateLookup(directories=[template_dir])
        directory_lookup.get_template('helpers.mako')

        for i, template_file in enumerate(template_files):
            log_debug("Creating file: {}", out_files[i])
            try:
                mytemplate = Template(filename=template_file, lookup=directory_lookup,
                                      imports=['import re'])
            except IOError:
                log_error(
                    "Could not find auto-generation code dependency: {}. Please make sure SNPE_ROOT"
                    " environment variable is set", template_file)
            except Exception as e:
                log_error('{} : {}', str(e), type(e))
                sys.exit(-1)
            log_debug_msg_as_status("Auto-generating code")
            rendered = mytemplate.render(**args)
            log_debug("Auto-generation complete")
            with open(out_files[i], 'wt+') as f:
                f.write(str(rendered).lstrip())


class UdoPackageInfo:
    """
    UdoPackageInfo contains information gleaned from the user provided config that will constitute a package.
    It is freely editable, meaning users can add and remove information as needed. It is also the main reference point
    for constructing a package.
    """

    def __init__(self, package_name, package_root, package_core_types, package_dsp_arch_types,
                 operators=None, snpe_udo_root=""):
        self.name = package_name
        self.root = os.path.join(package_root, package_name)
        self.core_types = package_core_types
        self.dsp_arch_types = package_dsp_arch_types
        self.operators = operators if operators else list()
        self.SNPE_UDO_ROOT = snpe_udo_root

    def get(self, attr, default=None):
        return getattr(self, attr, default)

    def add_operators(self, operators):
        for operator in operators:
            if isinstance(operator, Operator):
                self.operators.append(operator)
            else:
                raise TypeError('Operator must be a valid object of type {}'.format(
                    Operator.__class__.__name__))

    @staticmethod
    def from_dict(udo_package_dict):
        package_name = udo_package_dict.get("UDO_PACKAGE_NAME")
        root = os.path.abspath(udo_package_dict.get("UDO_PACKAGE_PATH", os.getcwd()))
        operators_list = udo_package_dict.get('Operators', list())
        operators = list(map(Operator.from_dict, operators_list))
        core_types = udo_package_dict.get("UDO_PACKAGE_CORETYPES",
                                          set(chain.from_iterable((operator_dict.get("core_types")
                                                                   for operator_dict in
                                                                   operators_list))))
        dsp_arch_types = udo_package_dict.get("UDO_PACKAGE_DSP_ARCH_TYPES",
                                              set(chain.from_iterable(
                                                  (operator_dict.get("dsp_arch_types", [])
                                                   for operator_dict in operators_list))))
        snpe_udo_root = os.environ.get('SNPE_UDO_ROOT', udo_package_dict.get('SNPE_UDO_ROOT', None))
        new_udo_package_info = UdoPackageInfo(package_name, root, list(core_types),
                                              list(dsp_arch_types), snpe_udo_root=snpe_udo_root)
        new_udo_package_info.add_operators(operators)

        return new_udo_package_info

    def __getattr__(self, item):
        return self.__getattribute__(item)

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def value_info(self):
        return {'package_name': self.root,
                'package_core_types': self.core_types,
                'operators': self.operators}


class UdoPackage:
    """
    The UdoPackage object is the core class used by the UdoGenerator and UdoFileGenerator objects. It contains a
    description of the package's operations, a catalog of op_names, their respective core_types and supported
    calculation types. Some of its members are intended to be set only once when a package is added,
    in contrast to the UdoPackageInfo. A package is expected to be created from a well defined package info only.

    """
    package_info = property_type('package_info', UdoPackageInfo)
    root = property_type('root', str)
    op_catalog_info = property_type('op_catalog_info', list)
    core_types = property_type('core_types', list)
    supported_runtimes = property_type('supported_runtimes',list)
    calculation_types = property_type('calculation_types', list)

    def __init__(self, package_name):
        self.name = package_name
        self.status = UdoPackageStatus.NOT_GENERATED

    def add_package_info(self, udo_package_info):
        """
        Add a package info to a package which formalizes and makes sure that relevant fields are mapped correctly to the
        SNPE_UDO API.
        :param udo_package_info: The information needed to define a package object.
        """
        self.package_info = udo_package_info
        self.name = self.package_info.get("udo_package_name", self.name)
        self.root = self.package_info.root
        self.op_catalog_info = [(str(operator['type_name']), operator['core_types']) for operator in
                                self.package_info.operators]
        self.core_types = get_internal_core_types(udo_package_info.core_types)
        self.dsp_arch_types = self.package_info.dsp_arch_types
        self.supported_runtimes = udo_package_info.core_types
        self.calculation_types = list(
            map(lambda x: SnpeUdoConstants.SNPE_CALCULATION_TYPES[x], udo_package_info.core_types))
