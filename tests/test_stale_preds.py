"""
Regression test: a proposal that was scored in an earlier stage and then
replaced by a shorter one during proposal generation must not stay in preds,
otherwise remove_proposals raises KeyError in ProposalGraph.remove_proposal.
"""

from collections import defaultdict
from types import SimpleNamespace

import pytest

from neuron_proofreader.proposal_graph import ProposalGraph
from neuron_proofreader.split_proofreading.split_inference import (
    LearnedSplitProofreader,
)


class StubGraph(ProposalGraph):
    def __init__(self):
        self.accepts = set()
        self.rejected = set()
        self.node_component_id = {1: 0, 2: 1, 3: 2}
        self.reset_proposals()

    def has_node(self, i):
        return True

    def degree(self, i):
        return 1

    def is_mergeable(self, i, j):
        return True


def make_proofreader(dataset, stage_log):
    pr = object.__new__(LearnedSplitProofreader)
    pr.dataset = dataset
    pr.log = lambda txt: None
    pr.save_connections = lambda: None

    def generate_proposals(config, radius, max_angle=None):
        if max_angle is not None:
            # Shorter candidate for the same component pair replaces {1, 2}
            dataset.remove_proposal(frozenset({1, 2}))
            dataset.add_proposal(1, 3)
            dataset.node_component_id[3] = 1
        return None

    def correct_splits(preds, dt, min_t, removal_t, suffix=""):
        stage_log.append(dict(preds))
        if not stage_log[1:]:
            return {frozenset({1, 2}): 0.1}
        pr.remove_proposals(preds, removal_t)
        return preds

    pr.generate_proposals = generate_proposals
    pr.correct_splits = correct_splits
    return pr


def test_replaced_proposal_does_not_crash():
    dataset = StubGraph()
    dataset.add_proposal(1, 2)
    config = SimpleNamespace(
        search_radii=lambda: [5.0],
        long_range_radius=10.0,
        long_range_angle=30.0,
    )
    stage_log = []
    pr = make_proofreader(dataset, stage_log)
    pr(config)

    # Stage 2 must only see scores for proposals still in the graph
    assert frozenset({1, 2}) not in stage_log[1]
