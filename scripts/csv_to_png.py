#!/usr/bin/env python3

from collections import defaultdict
import graphviz as gv

styles = {
    'graph': {
        'label': 'Operation Graph',
        'fontsize': '16',
        'fontcolor': 'white',
        'bgcolor': '#333333',
        'rankdir': 'BT',
    },
    'nodes': {
        'fontname': 'Helvetica',
        'shape': 'hexagon',
        'fontcolor': 'white',
        'color': 'white',
        'style': 'filled',
        'fillcolor': '#006699',
    },
    'edges': {
        'style': 'dashed',
        'color': 'white',
        'arrowhead': 'open',
        'fontname': 'Courier',
        'fontsize': '12',
        'fontcolor': 'white',
    }
}

def apply_styles(graph, styles):
    graph.graph_attr.update(
        ('graph' in styles and styles['graph']) or {}
    )
    graph.node_attr.update(
        ('nodes' in styles and styles['nodes']) or {}
    )
    graph.edge_attr.update(
        ('edges' in styles and styles['edges']) or {}
    )
    return graph

def str_clean(str):
    # get rid of <, >, |, :
    str = str.replace('<', '(')\
        .replace('>', ')')\
        .replace('|', '!')\
        .replace(':', '=')
    return str

def get_row_tuples(instream):
    return (tuple(col.strip() for col in line.split(',')) for line in instream)


def read_graph(instream):
    nodes = set()
    edges = defaultdict(list)
    for (observer, subject, order) in get_row_tuples(instream):
        nodes.add(str_clean(observer))
        nodes.add(str_clean(subject))
        edges[str_clean(observer)].append((str_clean(subject), order))
    return (nodes, edges)


def count_iterator(edges):
    return sum([len(v) for v in edges.values()])

def print_graph(callgraph):
    nodes, edges = callgraph

    g1 = gv.Digraph(format='png')
    for node in nodes:
        g1.node(node)

    for observer in edges:
        for subject, idx in edges[observer]:
            g1.edge(observer, subject, idx)

    apply_styles(g1, styles)
    g1.render('opgraph', view=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', nargs='?', default=None,
                        help='the CSV format callgraph to transform')
    args = parser.parse_args()

    import sys
    with (open(args.csv) if args.csv else sys.stdin) as infile:
        edgegraph = read_graph(infile)
    print_graph(edgegraph)

