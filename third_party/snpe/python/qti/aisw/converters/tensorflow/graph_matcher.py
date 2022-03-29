# =============================================================================
#
#  Copyright (c) 2016-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import itertools
import logging
from collections import Counter, OrderedDict, defaultdict

from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.tensorflow.util import ConverterError


class IGraphNode(object):
    def __init__(self, identifier, node_types, original_node, inputs, should_link_inputs):
        self.identifier = identifier
        self.node_types = node_types
        self.original_node = original_node
        self.should_link_inputs = should_link_inputs
        if inputs is not None:
            self.inputs = inputs
        else:
            self.inputs = []

    @property
    def is_consumable(self):
        return True


class ConverterRepeatableSequenceTreeNode(IGraphNode):
    """
    This node type is able to describe a repeatable cell_sequence of unknown size.

    current limitation: cannot be last output/ one output node one input node
    """
    def __init__(self, identifier, tree_output_node, tree_input_node, inputs=None):
        super(ConverterRepeatableSequenceTreeNode, self).__init__(identifier, node_types=tree_output_node.node_types,
                                                                  original_node=self, inputs=inputs,
                                                                  should_link_inputs=False)
        self.tree_output_node = tree_output_node
        self.tree_input_node = tree_input_node

    def create_repeatable_sequence_outputs(self, repeatable_sequence_count):
        output_nodes = []
        for i in range(0, repeatable_sequence_count):
            index_of_repetition = '_' + str(i + 1)
            sequence_output = self._create_sequence_ops(index_of_repetition)
            output_nodes.append(sequence_output)
        return output_nodes

    def _create_sequence_ops(self, index_of_repetition):
        new_output_node = ConverterSequenceNode(self.tree_output_node.identifier + index_of_repetition,
                                                self.tree_output_node.node_types,
                                                inputs=[])
        nodes_to_be_replicated = [(self.tree_output_node, new_output_node)]
        while nodes_to_be_replicated:
            original_node, replicated_node = nodes_to_be_replicated.pop()
            for input_node in original_node.inputs:
                child_node = ConverterSequenceNode(input_node.identifier + index_of_repetition,
                                                   input_node.node_types,
                                                   inputs=[])
                replicated_node.inputs.append(child_node)
                nodes_to_be_replicated.append((input_node, child_node))
            if original_node == self.tree_input_node:
                replicated_node.inputs.extend(self.inputs)
        return new_output_node


class ConverterSequenceNode(IGraphNode):
    """
    This node type defines a node in a cell_sequence definition.
    """
    def __init__(self, identifier, node_types, inputs=None):
        super(ConverterSequenceNode, self).__init__(identifier, node_types, original_node=self, inputs=inputs,
                                                    should_link_inputs=False)


class NonConsumableConverterSequenceNode(IGraphNode):
    """
    This node type defines a node in a cell_sequence definition which is not consumed
    as part of a cell_sequence.
    """
    def __init__(self, identifier, node_types, inputs=None):
        super(NonConsumableConverterSequenceNode, self).__init__(identifier, node_types, original_node=self,
                                                                 inputs=inputs, should_link_inputs=False)
    @property
    def is_consumable(self):
        return False


class TFOperationNode(IGraphNode):
    def __init__(self, tf_op, should_link_inputs=True):
        super(TFOperationNode, self).__init__(tf_op.name, [tf_op.type], original_node=tf_op, inputs=[],
                                              should_link_inputs=should_link_inputs)


class NonConsumableTFOperationNode(IGraphNode):
    def __init__(self, tf_op, should_link_inputs=True):
        super(NonConsumableTFOperationNode, self).__init__(tf_op.name, [tf_op.type], original_node=tf_op, inputs=[],
                                                           should_link_inputs=should_link_inputs)
    @property
    def is_consumable(self):
        return False


class TFGraphBuilder(object):

    def __init__(self, tf_ops):
        self.nodes_map = OrderedDict()
        for tf_op in tf_ops:
            node = TFOperationNode(tf_op)
            self.nodes_map[node.identifier] = node

    def link_nodes(self):
        for node in list(self.nodes_map.values()):
            if not node.should_link_inputs:
                continue
            self._link_node_inputs(node)
            node.should_link_inputs = False

    @property
    def nodes(self):
        return list(self.nodes_map.values())

    def _link_node_inputs(self, node):
        for tf_tensor in node.original_node.inputs:
            input_node = self.nodes_map.get(tf_tensor.op.name, None)
            if input_node is None:
                input_node = NonConsumableTFOperationNode(tf_tensor.op)
                self.nodes_map[tf_tensor.op.name] = input_node
            node.inputs.append(input_node)


class GraphMatch(OrderedDict):
    def __init__(self, seq_to_node_map, sequence):
        """
        :type seq_to_node_map: dict(IGraphNode, IGraphNode)
        :type sequence: GraphSequence
        """
        super(GraphMatch, self).__init__()
        self._id_to_seq_map = OrderedDict()
        self._id_to_node_map = OrderedDict()
        ordered_nodes = list(sequence.values())

        for seq_node in list(sequence.values()):
            if seq_node not in seq_to_node_map:
                ordered_nodes.remove(seq_node)

        for seq_node in list(seq_to_node_map.keys()):
            if seq_node not in ordered_nodes:
                ordered_nodes.append(seq_node)

        for seq_node in ordered_nodes:
            graph_node = seq_to_node_map[seq_node]
            self._id_to_seq_map[seq_node.identifier] = seq_node
            self._id_to_node_map[seq_node.identifier] = graph_node
            self[seq_node.identifier] = graph_node.original_node

    @property
    def consumed_nodes(self):
        consumed_nodes = []
        for seq_id, original_node in list(self.items()):
            if self._id_to_seq_map[seq_id].is_consumable and self._id_to_node_map[seq_id].is_consumable:
                consumed_nodes.append(original_node)
        return consumed_nodes


class GraphSequence(OrderedDict):
    def __init__(self, nodes=None):
        if nodes is None:
            nodes = []
        super(GraphSequence, self).__init__()
        self._output_nodes = []
        for node in nodes:
            if node.identifier in self:
                raise ConverterError('Node with id already defined {}'.format(node.identifier))
            self[node.identifier] = node

    def set_inputs(self, node_id, input_ids):
        target_node = self[node_id]
        for input_id in input_ids:
            target_node.inputs.append(self[input_id])

    def set_outputs(self, output_ids):
        for output_id in output_ids:
            self._output_nodes.append(self[output_id])

    def clear_inputs_for_nodes(self, node_ids):
        for node_id in node_ids:
            if node_id not in self:
                raise ConverterError('Unable to clear inputs for node_id[{}]: Node not found.'.format(node_id))
            target_node = self[node_id]
            del target_node.inputs[:]

    def clear_outputs(self):
        self._output_nodes = []

    @property
    def output_nodes(self):
        return self._output_nodes[:]


class GraphMatcher(object):
    def __init__(self, graph, graph_helper):
        """
        :type graph: list(IGraphNode)
        """
        self.graph = list(graph)
        self._graph_helper = graph_helper
        self.consumed_nodes = []
        self._evaluated_tensors = set()

    def _check_and_add_tensor(self, batch_tensors, batch_tensors_by_type, tensor):
        """
        :type batch_tensors: set(tensorflow.Tensor)
        :type batch_tensors_by_type: dict(tensorflow.Tensor.op.type, tensorflow.Tensor)
        :type tensor: tensorflow.Tensor
        :rtype: None
        """
        if tensor not in self._evaluated_tensors:
            batch_tensors_by_type[tensor.op.type].append(tensor)
            batch_tensors.append(tensor)
            self._evaluated_tensors.add(tensor)

    def _evaluate_tensors(self, matches):
        """
        The function will batch evaluate the tensors which will be evaluated later of each GraphMatch.
        The batch evaluation is more efficient and the results will cache in graph_helper.

        :type matches: list(GraphMatch)
        :rtype: None
        """

        batch_tensors = []
        batch_tensors_by_type = defaultdict(list)
        graph_helper = self._graph_helper
        for match in matches:
            for op_name in match:
                tensor = graph_helper._graph.get_tensor_by_name(
                    graph_helper.indexed_tensor_name(match[op_name].name, 0)
                )
                self._check_and_add_tensor(batch_tensors, batch_tensors_by_type, tensor)

                for input_tensor in match[op_name].inputs:
                    self._check_and_add_tensor(batch_tensors, batch_tensors_by_type, input_tensor)

                    input_, _ = graph_helper.get_none_identity_input(input_tensor)
                    self._check_and_add_tensor(batch_tensors, batch_tensors_by_type, input_)

                for output_tensor in match[op_name].outputs:
                    self._check_and_add_tensor(batch_tensors, batch_tensors_by_type, output_tensor)

        try:
            # Evaluate the whole tensors and cache the result,
            # which will be heavily requested from resolvers later.
            graph_helper.evaluate_tensors_output(batch_tensors)
        except Exception:
            # Some types, e.g. Merge, LoopCond or TensorArrayV3,
            # are not able to evaluate, and causes the TensorFlow InternalError.
            # Therefore, for each type to evaluate the tensors.
            for op_type in batch_tensors_by_type:
                try:
                    graph_helper.evaluate_tensors_output(batch_tensors_by_type[op_type])
                except Exception:
                    logger = logging.getLogger()
                    logger.debug(code_to_message.get_error_message('ERROR_TF_FALLBACK_TO_ONDEMAND_EVALUATION'))

    def match_sequence(self, sequence):
        """
        :type sequence: GraphSequence
        :rtype: list(GraphMatch)
        """
        self.consumed_nodes = []
        mappings = self._match_sequence_from_roots(sequence.output_nodes)
        matches = [GraphMatch(mapping, sequence) for mapping in mappings]
        # Matches can be found even if the list of consumed nodes is empty (this occurs when a root node is
        # Non-consumable). If consumed nodes is empty, we should return an empty list.
        filtered_matches = [match for match in matches if match.consumed_nodes]

        # Batch evaluate the tensors, which will be heavily requested from resolvers later,
        # and cache the results.
        self._evaluate_tensors(filtered_matches)
        return filtered_matches

    def _generate_roots_candidate_assignments(self, sequence_graph_roots):
        candidate_roots_in_graph = []
        for _ in sequence_graph_roots:
            candidate_roots_in_graph.append([])
        for graph_node in self.graph:
            for root_index in range(0, len(sequence_graph_roots)):
                if self._match_one_node_type(sequence_graph_roots[root_index], graph_node):
                    candidate_roots_in_graph[root_index].append(graph_node)
        # Using generator for reducing memory consumption for huge graph like nms
        for candidate in itertools.product(*candidate_roots_in_graph):
            if len(set(candidate)) == len(sequence_graph_roots):
                yield candidate

    def _match_sequence_from_roots(self, sequence_graph_roots):
        """
        :param sequence_graph_roots: list[IGraphNode]
        :return: list(dict(IGraphNode, IGraphNode))
        """
        matches_assignments = []
        for roots_assignment in self._generate_roots_candidate_assignments(sequence_graph_roots):
            roots_assignment_map = dict()
            # map roots_assignment
            not_visited_queue = []
            for root_index in range(0, len(roots_assignment)):
                sequence_root = sequence_graph_roots[root_index]
                roots_assignment_map[sequence_root] = roots_assignment[root_index]
                not_visited_queue.append(sequence_root)
            # start exploring with the roots_assignment mapped
            match_assignments = self._match_next_level_with_assignments(not_visited_queue, roots_assignment_map)
            if match_assignments is not None:
                # remove all consumed nodes for next iteration
                matches_assignments.append(match_assignments)
                self._remove_consumed_nodes_for_next_iteration(match_assignments)
        return matches_assignments

    def _remove_consumed_nodes_for_next_iteration(self, match_assignments):
        for sequence_node, matched_node in list(match_assignments.items()):
            if sequence_node.is_consumable:
                self.consumed_nodes.append(matched_node)

    def _match_next_level_with_assignments(self, not_visited_queue, current_assignments):
        if len(not_visited_queue) == 0:
            return current_assignments

        next_sequence_node = not_visited_queue.pop(0)
        if len(next_sequence_node.inputs) == 0:
            return self._match_next_level_with_assignments(not_visited_queue, current_assignments)

        candidate_node = current_assignments[next_sequence_node]
        # check if types as set will match, then get 1-1 correspondance
        next_level_assignments = self._match_nodes_types(next_sequence_node.inputs, candidate_node.inputs)
        next_level_assignments = self._filter_invalid_candidate_assignments(next_level_assignments,
                                                                            current_assignments)
        for next_level_assignment in next_level_assignments:
            next_level_visited_queue = not_visited_queue[:]
            next_level_visited_queue.extend(set(next_level_assignment) - set(current_assignments))
            next_level_assignments = current_assignments.copy()
            next_level_assignments.update(next_level_assignment)
            result = self._match_next_level_with_assignments(next_level_visited_queue, next_level_assignments)
            if result is not None:
                return result
        return None

    def _filter_invalid_candidate_assignments(self, candidate_assignments, current_assignments):
        valid_assignments = []
        for assignment in candidate_assignments:
            valid_assignments.append(assignment)
            for sequence_node, graph_node in list(assignment.items()):
                already_consumed = (sequence_node.is_consumable and graph_node in self.consumed_nodes)
                conflicting_assignment = (sequence_node in list(current_assignments.keys()) and
                                          graph_node != current_assignments[sequence_node])
                if already_consumed or conflicting_assignment:
                    valid_assignments.pop()
                    break
        return valid_assignments

    def _match_nodes_types(self, sequence_nodes, graph_nodes):
        if self._sequence_contains_repeatable_nodes(sequence_nodes):
            sequence_nodes = self._prepare_nodes_list_for_repeatable_sequence(sequence_nodes, graph_nodes)
        if len(sequence_nodes) != len(graph_nodes):
            return []

        nodes_matches = self._create_nodes_candidates_lists(sequence_nodes, graph_nodes)
        assignments = self._create_candidate_assignments(nodes_matches, sequence_nodes)
        return assignments

    @classmethod
    def _sequence_contains_repeatable_nodes(cls, sequence_nodes):
        trigger_repeatable_tree_matching = False
        for node in sequence_nodes:
            if isinstance(node, ConverterRepeatableSequenceTreeNode):
                trigger_repeatable_tree_matching = True
                break
        return trigger_repeatable_tree_matching

    @classmethod
    def _create_candidate_assignments(cls, matches_list, sequence_nodes):
        matches_maps_list = []
        for matches in matches_list:
            # remove combinations that has duplicates
            if len(set(sequence_nodes)) != len(set(matches)):
                continue
            matches_map = dict()
            for nodeA_index in range(0, len(sequence_nodes)):
                matches_map[sequence_nodes[nodeA_index]] = matches[nodeA_index]
            matches_maps_list.append(matches_map)
        return matches_maps_list

    def _create_nodes_candidates_lists(self, sequence_nodes, graph_nodes):
        matches = []
        for _ in sequence_nodes:
            matches.append([])
        for nodeA_index in range(0, len(sequence_nodes)):
            for nodeB in graph_nodes:
                if '?' in sequence_nodes[nodeA_index].node_types:
                    matches[nodeA_index].append(nodeB)
                elif self._match_one_node_type(sequence_nodes[nodeA_index], nodeB):
                    matches[nodeA_index].append(nodeB)
        return list(itertools.product(*matches))

    @classmethod
    def _match_one_node_type(cls, node1, node2):
        for nodeA_type in node1.node_types:
            for nodeB_type in node2.node_types:
                if nodeA_type.lower() == nodeB_type.lower():
                    return True
        return False

    @classmethod
    def _prepare_nodes_list_for_repeatable_sequence(cls, sequence_nodes, graph_nodes):
        graph_nodes_types = []
        for node in graph_nodes:
            graph_nodes_types.extend(node.node_types)

        sequence_nodes_types = []
        repeatable_nodes = []
        for node in sequence_nodes:
            if not isinstance(node, ConverterRepeatableSequenceTreeNode):
                sequence_nodes_types.extend(node.node_types)
            else:
                repeatable_nodes.append(node)

        diff_between_children = list((Counter(graph_nodes_types) - Counter(sequence_nodes_types)).elements())
        if not len(set(diff_between_children)) == 1:
            return []

        expanded_sequence_nodes = list(sequence_nodes)
        for repeatable_node in repeatable_nodes:
            expanded_sequence_nodes.remove(repeatable_node)
            outputs = repeatable_node.create_repeatable_sequence_outputs(len(diff_between_children))
            expanded_sequence_nodes.extend(outputs)
        return expanded_sequence_nodes
