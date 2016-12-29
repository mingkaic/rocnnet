//
// Created by Mingkai Chen on 2016-12-27.
//

#pragma once
#ifndef creator_vertex_hpp
#define creator_vertex_hpp

// purpose: hide all tenncor implementation

namespace tensorio
{

// connector types
enum CONNECTOR_MAP
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

// encapsulate leaf building options
// defaults to random initialization between -1 and 1
struct var_opt
{
	std::vector<size_t> shape = {1};
	LEAF_MAP get_type (void) = 0;
};
struct place_opt : public var_opt
{
	LEAF_MAP get_type (void) { return PLACE; }
};
struct const_opt : public var_opt
{
	double val_ = 0;
	LEAF_MAP get_type (void) { return CONST; }
};
struct rand_opt : public var_opt
{
	double min_ = -1;
	double max_ = 1;
	LEAF_MAP get_type (void) { return RAND; }
};

// store and retrieve graph information
// builder of graph
class vertex_manager
{
	private:
		enum NODE_TYPE;
		struct node_registry;

		node_registry inst;

	public:
		// MODIFIERS
		// register nodes
		std::string register_op (CONNECTOR_MAP cm);
		std::string register_leaf (std::string label, var_opt opt = rand_opt());
		// delete nodes
		bool delete_node (std::string id);
		// link id1 to id2 if id2 points to a connector
		void link_nodes (std::string id1, std::string id2, size_t index = 0);
		// delete links between id1 and id2 (directionality does not matter)
		bool delete_link (std::string id1, std::string id2);
		// FORWARD MODE ACCESSORS

		// BACKWARD MODE ACCESSORS
};

}

#endif /* creator_vertex_hpp */
