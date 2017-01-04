//
// Created by Mingkai Chen on 2016-12-27.
//

#include <vector>
#include <string>
#include <unordered_set>
#include <utility>
#include <experimental/optional>

#pragma once
#ifndef creator_vertex_hpp
#define creator_vertex_hpp

// purpose: hide all tenncor implementation

namespace tensorio
{

// connector types
enum CONNECTOR_TYPE
{
	// unaries
	ABS,
	NEG,
	SIN,
	COS,
	TAN,
	CSC,
	SEC,
	COT,
	EXP,
	// scalar operations
//	SQRT, POW, CLIP_VAL, CLIP_NORM, EXTEND, COMPRESS
	// binaries
	ADD,
	SUB,
	MUL,
	DIV,
	// transformations
	MATMUL,
	TRANS,
	FIT,
};

// leaf types
enum LEAF_TYPE
{
	PLACE,
	CONST, // still uses a variable (sets an initial value)
	RAND
};

union var_param
{
	double val_;
	std::pair<double,double> min2max_;
};

// encapsulate leaf building options
// defaults to random initialization between -1 and 1
struct var_opt
{
	LEAF_TYPE type = PLACE;
	std::vector<size_t> shape_ = {1};
	std::experimental::optional<var_param> parameter_;
};

// from are children, to are parents
struct connection
{
	std::string from_id;
	std::string to_id;
	size_t idx;
};

struct connection_hash
{
	size_t operator() (const connection& conn) const
	{
		std::hash<std::string> hash_fn;
		return hash_fn(conn.from_id) * hash_fn(conn.to_id) * (conn.idx + 1);
	}
};

inline bool operator == (const connection& lhs, const connection& rhs)
{
	connection_hash hash_fn;
	return hash_fn(lhs) == hash_fn(rhs);
}

using CONNECTION_SET = std::unordered_set<connection, connection_hash>;

struct metainfo
{
	// no value means it's a leaf
	std::experimental::optional<CONNECTOR_TYPE> op_type_;
};

// store and retrieve graph information
// builder of graph
class vertex_manager
{
	private:
		struct node_registry;

		node_registry* inst;

	public:
		vertex_manager (void);
		virtual ~vertex_manager (void);

		// MODIFIERS
		// register nodes
		std::string register_op (CONNECTOR_TYPE cm);
		std::string register_leaf (std::string label, var_opt opt);
		// delete nodes
		bool delete_node (std::string id);
		// link from_id to to_id if to_id points to a connector.
		// index denotes link's index to to_id
		void link (std::string from_id, std::string to_id, size_t index = 0);
		// delete indexed link to id node
		bool delete_link (std::string id, size_t index);

		// ACCESSORS
		// return no value if id is not found
		std::experimental::optional<metainfo> node_info (std::string id) const;
		// grab all connections under sub-network flow connected to root (breadth first search)
		void get_connections (CONNECTION_SET& conns, std::string root_id) const;
		// get forward graph vertices (ids) and edges (conns)
		void get_forwards (std::unordered_set<std::string>& ids, CONNECTION_SET& conns) const;
		// get backward (gradient) graph vertices (ids) and edges (conns)
		void get_backwards (std::unordered_set<std::string>& ids, CONNECTION_SET& conns) const;
};

}

#endif /* creator_vertex_hpp */
