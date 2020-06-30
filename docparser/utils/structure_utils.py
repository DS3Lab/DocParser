import logging
from collections import defaultdict

logger = logging.getLogger(__name__)
import networkx as nx

from docparser.objdetmetrics_lib.utils import BBFormat


def generate_new_anns_from_bbox_object(bbox_object, parent_id, max_id, page=0):
    ann_id = bbox_object.getBboxID()
    category = bbox_object.getClassId()
    bbox_xywh = bbox_object.getAbsoluteBoundingBox(format=BBFormat.XYWH)
    new_ann = {'id': ann_id, 'parent': parent_id, 'category': category}
    bbox_ann_child = {'id': max_id + 1, 'parent': ann_id, 'category': 'box', 'bbox': bbox_xywh, 'page': page}
    max_id = max_id + 1
    return new_ann, bbox_ann_child, max_id


def create_annotations_from_bbox_ids(sorted_node_ids, bbox_by_id, parent_id, max_id):
    new_annotations = []
    for node_id in sorted_node_ids:
        bbox_object = bbox_by_id[node_id]
        new_ann, new_bbox_ann, max_id = generate_new_anns_from_bbox_object(bbox_object, parent_id, max_id)
        new_annotations += [new_ann, new_bbox_ann]
    return new_annotations, max_id


def get_sorted_sequence_of_nodes(doc_graph, sibling_ids):
    sibling_subgraph = doc_graph.subgraph(sibling_ids)

    if len(sibling_ids) <= 1:
        return list(sibling_ids)

    root_nodes = [node for node, in_degree in sibling_subgraph.in_degree() if
                  in_degree == 0]  # root node: node which has no incoming edges
    if len(root_nodes) != 1:
        raise AssertionError(
            "nodes with in_degree 0: {}, all sibling nodes passed: {}, subgraph nodes: {}, subgraph edges: {}".format(
                root_nodes, sibling_ids, list(sibling_subgraph.nodes()), list(sibling_subgraph.edges())))

    root_node = root_nodes[0]

    sorted_siblings = list(nx.topological_sort(sibling_subgraph))
    assert sorted_siblings[0] == root_node
    return sorted_siblings


def create_flat_annotation_list(bboxes, meta_bboxes, hierarchy_relations, sequence_relations):
    max_id = max(bbox.getBboxID() for bbox in bboxes)
    bbox_by_id = {bbox.getBboxID(): bbox for bbox in bboxes}
    document_root_id = max_id + 1
    meta_root_id = max_id + 2
    other_root_id = max_id + 3
    max_id = max_id + 4
    all_annotations = [
        {"id": other_root_id, "category": "unk", "parent": None},
        {"id": meta_root_id, "category": "meta", "parent": None},
        {"id": document_root_id, "category": "document", "parent": None}
    ]

    # add all meta annotations
    meta_ids = set([x.getBboxID() for x in meta_bboxes])
    new_annotations, max_id = create_annotations_from_bbox_ids(meta_ids, bbox_by_id, meta_root_id, max_id)
    all_annotations += new_annotations

    children_by_parent = defaultdict(set)

    doc_graph = nx.OrderedDiGraph()
    all_bbox_ids = set(bbox.getBboxID() for bbox in bboxes)
    all_non_meta_bbox_ids = all_bbox_ids - meta_ids
    all_ids_occurring_in_rels = set()
    all_ids_occurring_in_sequence_rels = set()
    for (subj, obj, rel) in hierarchy_relations:
        children_by_parent[subj].add(obj)
        all_ids_occurring_in_rels.add(subj)
        all_ids_occurring_in_rels.add(obj)

    for (subj, obj, rel) in sequence_relations:
        doc_graph.add_edge(subj, obj)
        all_ids_occurring_in_sequence_rels.add(subj)
        all_ids_occurring_in_sequence_rels.add(obj)

    all_ids_occurring_in_rels = all_ids_occurring_in_rels.union(all_ids_occurring_in_sequence_rels)
    logger.debug("{}/{} size for all bbox/relation ids: {}, {}".format(len(all_non_meta_bbox_ids),
                                                                       len(all_ids_occurring_in_rels),
                                                                       all_non_meta_bbox_ids,
                                                                       all_ids_occurring_in_rels))

    # if there's at least two 'normal' annotations, there must exist a relation that covers both of them
    if len(
            all_non_meta_bbox_ids) > 1:
        assert all_ids_occurring_in_rels == all_non_meta_bbox_ids

    non_children_ids = set()
    non_children_ids = non_children_ids.union(all_non_meta_bbox_ids)
    for parent_id, children_ids in children_by_parent.items():
        non_children_ids = non_children_ids - children_ids

    # TODO: deal with nodes on same level that were not found to have a sequence relation. append them to 'unk' element?
    sorted_toplevel_node_ids = get_sorted_sequence_of_nodes(doc_graph, non_children_ids)

    new_annotations, max_id = create_annotations_from_bbox_ids(sorted_toplevel_node_ids, bbox_by_id, document_root_id,
                                                               max_id)
    all_annotations += new_annotations
    for parent_id, children_ids in children_by_parent.items():
        if len(children_ids) > 1:
            assert set(children_ids).issubset(all_ids_occurring_in_sequence_rels)
        sorted_sibling_sequence = get_sorted_sequence_of_nodes(doc_graph, children_ids)
        new_annotations, max_id = create_annotations_from_bbox_ids(sorted_sibling_sequence, bbox_by_id, parent_id,
                                                                   max_id)
        all_annotations += new_annotations

    all_non_box_ann_ids = set(
        ann['id'] for ann in all_annotations if ann['parent'] is not None and ann['category'] != 'box')
    logger.debug("Generated annotation ids: {}, original bbox ids: {}".format(all_non_box_ann_ids, all_bbox_ids))
    assert all_non_box_ann_ids == all_bbox_ids

    return all_annotations
