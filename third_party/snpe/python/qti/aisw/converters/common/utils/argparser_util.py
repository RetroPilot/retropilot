# ==============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import copy
import argparse
import textwrap as _textwrap


class CustomHelpFormatter(argparse.HelpFormatter):
    def __init__(self,
                 prog,
                 indent_increment=2,
                 max_help_position=24,
                 width=None):
        super(CustomHelpFormatter, self).__init__(prog, indent_increment, max_help_position, width=100)

    def _split_lines(self, text, width):
        # Preserve newline character in the help text
        paras = text.splitlines()
        lines = []
        for para in paras:
            # Wrap the paragraphs based on width
            lines.extend(_textwrap.wrap(para, width, replace_whitespace=False))
        return lines


class ArgParserWrapper(object):
    """
    Wrapper class for argument parsing
    """

    def __init__(self, parents=[], **kwargs):
        self.parser = argparse.ArgumentParser(add_help=False, **kwargs)
        self.argument_groups = {}

        self.required = self.add_argument_group('required arguments')
        self.optional = self.parser.add_argument_group('optional arguments')
        self.optional.add_argument("-h", "--help", action="help", help="show this help message and exit")

        self._extend_from_parents(parents)

    def _extend_from_parents(self, parents):
        def _remove_action_from_main_group(main_group_, action_):
            for a in main_group_._actions:
                if a.dest == action_.dest and a in main_group_._group_actions:
                    main_group_._remove_action(a)
                    for optstr in a.option_strings:
                        del main_group_._option_string_actions[optstr]

        for i, parent in enumerate(parents):
            if not isinstance(parent, ArgParserWrapper):
                raise TypeError("Parent {0} not of Type ArgParserWrapper".format(parent.__class__.__name__))
            for action in parent.required._group_actions:
                self.required._add_action(action)
            for action in parent.optional._group_actions:
                self.optional._add_action(action)
            for group in parent.parser._action_groups:
                if group.title in ['required arguments', 'optional arguments']:
                    continue
                if group.title in self.argument_groups:
                    new_group = self.argument_groups[group.title]
                else:
                    new_group = self.parser.add_argument_group(group.title)
                    self.argument_groups[group.title] = new_group
                for action in group._group_actions:
                    new_group._add_action(copy.copy(action))
            for group in parent.parser._mutually_exclusive_groups:
                group_title = getattr(group._container, 'title', None)
                if group_title in self.argument_groups:
                    main_group = self.argument_groups[group_title]
                else:
                    main_group = self.optional

                me_group = main_group.add_mutually_exclusive_group()
                for action in group._group_actions:
                    # If mutually exclusive arguments are part of non-default group(i.e defined by us) then
                    # they will need to be added/tagged as a mutually exlusive group subset of the main group.
                    # Hence we first need to remove from top-level of the group then re-add to the me_group subset
                    _remove_action_from_main_group(main_group, action)
                    me_group._add_action(copy.copy(action))

            # Add epilog
            existing_epilog = getattr(self.parser, 'epilog')
            if existing_epilog is None:
                existing_epilog = ""
            epilog = getattr(parent.parser, 'epilog')
            if epilog is not None:
                setattr(self.parser, 'epilog', existing_epilog + '\n' + epilog)

    def add_required_argument(self, *args, **kwargs):
        self.required.add_argument(*args, required=True, **kwargs)

    def add_optional_argument(self, *args, **kwargs):
        self.optional.add_argument(*args, required=False, **kwargs)

    def add_argument_group(self, title, *args, **kwargs):
        if title in self.argument_groups:
            return self.argument_groups[title]
        new_group = self.parser.add_argument_group(title, *args, **kwargs)
        self.argument_groups[title] = new_group
        return new_group

    def add_mutually_exclusive_args(self, *args: list):
        # confine to set
        args_as_set = set(args)

        # Add epilog note
        existing_epilog = getattr(self.parser, 'epilog')
        if existing_epilog is None:
            existing_epilog = ""
        exclusivity_info = "Note: Only one of: {} can be specified\n".format(str(args_as_set))
        setattr(self.parser, 'epilog', existing_epilog + exclusivity_info)

    def parse_args(self, args=None, namespace=None):
        cmd_args = self.parser.parse_args(args, namespace)
        return cmd_args
