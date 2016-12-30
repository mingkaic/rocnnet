//
// Created by Mingkai Chen on 2016-12-27.
//

#include <vector>
#include <string>
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
enum LEAF_MAP
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
	LEAF_MAP type = PLACE;
	std::vector<size_t> shape_ = {1};
	std::experimental::optional<var_param> parameter_;
};

// from are children, to are parents
struct connection
{
	std::string from_id;
	std::string to_id;
};

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
		~vertex_manager (void);

		// MODIFIERS
		// register nodes
		std::string register_op (CONNECTOR_TYPE cm);
		std::string register_leaf (std::string label, var_opt opt);
		// delete nodes
		bool delete_node (std::string id);
		// link id1 to id2 if id2 points to a connector.
		// index denotes link's index to id2
		void link (std::string id1, std::string id2, size_t index = 0);
		// delete indexed link to id node
		bool delete_link (std::string id, size_t index);

		// COMMON ACCESSORS
		// return no value if id is not found
		std::experimental::optional<metainfo> node_info (std::string id);
		void get_connections (
				std::vector<connection>& conns,
				std::string root_id);
		// FORWARD MODE ACCESSORS
		void get_forwards (std::vector<std::string>& ids);
		// BACKWARD MODE ACCESSORS
		void get_backwards (std::vector<std::string>& ids);
};

}

#endif /* creator_vertex_hpp */
