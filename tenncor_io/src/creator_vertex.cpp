//
// Created by Mingkai Chen on 2016-12-27.
//

#include "graph/mutable/mutable_connector.hpp"
#include "creator_vertex.hpp"

#ifdef creator_vertex_hpp

namespace tensorio
{

// adhoc function mapping connector types to connector constructors
static nnet::mutable_connector<double>* mutable_build (CONNECTOR_MAP cm)
{
	size_t nargs = 0;
	nnet::MAKE_CONNECT<double> maker;
	switch (cm)
	{
		case ABS:
			nargs = 1;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return +args[0];
			};
			break;
		case NEG:
			nargs = 1;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return -args[0];
			};
			break;
		case SIN:
			nargs = 1;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return nnet::sin(args[0]);
			};
			break;
		case COS:
			nargs = 1;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return nnet::cos(args[0]);
			};
			break;
		case TAN:
			nargs = 1;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return nnet::tan(args[0]);
			};
			break;
		case CSC:
			nargs = 1;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return nnet::csc(args[0]);
			};
			break;
		case SEC:
			nargs = 1;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return nnet::sec(args[0]);
			};
			break;
		case COT:
			nargs = 1;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return nnet::cot(args[0]);
			};
			break;
		case EXP:
			nargs = 1;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return nnet::exp(args[0]);
			};
			break;
		case ADD:
			nargs = 2;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return args[0] + args[1];
			};
			break;
		case SUB:
			nargs = 2;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return args[0] - args[1];
			};
			break;
		case MUL:
			nargs = 2;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return args[0] * args[1];
			};
			break;
		case DIV:
			nargs = 2;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return args[0] / args[1];
			};
			break;
		case MATMUL:
			nargs = 2;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return nnet::matmul(args[0], args[1]);
			};
			break;
		case FIT:
			nargs = 2;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return nnet::fit(args[0], args[1]);
			};
			break;
		case TRANS:
			nargs = 1;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return nnet::transpose(args[0]);
			};
			break;
		default:
			return nullptr;
	}
	return nnet::mutable_connector<double>::build(maker, nargs);
}

enum vertex_manager::NODE_TYPE
{
	OPERATION,
	LEAF,
	NONE
};

struct vertex_manager::node_registry
{
	std::unordered_map<std::string,nnet::mutable_connector<double>*> ops_;
	std::unordered_map<std::string,nnet::ileaf<double>*> leaves_;

	// check if id exists in ops_ or leaves_, extract the node, then return its type
	NODE_TYPE grab_node (ivariable<double>*& out, std::string id)
	{
		auto it = ops_.find(id1);
		if (ops_.end() == it)
		{
			out = *it;
			return OPERATION;
		}
		else if (ops.end() == (it = leaves_.find(id1)))
		{
			out = *it;
			return LEAF;
		}
		// error not found
		return NONE;
	}
};

NODE_TYPE vertex_manager::grab_node (ivariable<double>*& out, std::string id)
{
	auto it = inst.ops_.find(id1);
	if (inst.ops_.end() == it)
	{
		out = *it;
		return OPERATION;
	}
	else if (ops.end() == (it = inst.leaves_.find(id1)))
	{
		out = *it;
		return LEAF;
	}
	// error not found
	return NONE;
}

std::string vertex_manager::register_op (CONNECTOR_MAP cm)
{
	nnet::mutable_connector<double>* con = mutable_build(cm);
	std::string id = con->get_uid();
	inst.ops_[id] = con;
	return id;
}

std::string vertex_manager::register_leaf (std::string label, var_opt opt)
{
	nnet::ileaf<double>* leaf = nullptr;
	switch (opt.get_type())
	{
		case PLACE:
			leaf = new nnet::placeholder(opt.shape, label);
			break;
		case CONST:
			const_op* op = static_cast<const_op*>(&opt);
			nnet::const_init<double> init(op->val_);
			leaf = new nnet::variable(opt.shape, init, label);
			break;
		case RAND:
			rand_op* op = static_cast<rand_op*>(&opt);
			nnet::random_uniform<double> init(op->min_, op->max_);
			leaf = new nnet::variable(opt.shape, init, label);
			break;
	}
	std::string id = leaf->get_uid();
	inst.leaves_[id] = leaf;
	return id;
}

bool vertex_manager::delete_node (std::string id)
{
	ivariable<double>* ele = nullptr;
	NODE_TYPE type = inst.grab_node(ele, id);
	if (LEAF == type)
	{
		inst.leaves_.erase(id);
	}
	else if (OPERATION == type)
	{
		inst.ops_.erase(id);
	}
	if (nullptr != ele)
	{
		// unattach permanent node from mutable_connector wrapper
		ccoms::update_message msg;
		msg.cmd_ = REMOVE_ARG;
		ele.notify(msg);
		// delete ele: triggering chain destruction
		delete ele;
	}
}

void vertex_manager::link_nodes (std::string id1, std::string id2, size_t index)
{
	ivariable<double>* ele1 = nullptr;
	ivariable<double>* ele2 = nullptr;
	NODE_TYPE t1 = inst.grab_node(ele1, id1);
	NODE_TYPE t2 = inst.grab_node(ele2, id2);

	// error: one non-existent node
	if (NONE == t1 || NONE == t2)
	{
		std::string invalid_id = NONE == t1 ? id1 : id2;
		throw std::invalid_argument("invalid id: (" + invalid_id + ") node not found");
	}
	// error: can't connect to a leaf
	if (LEAF == t2)
	{
		throw std::invalid_argument("can't connect to leaf " + elem2->get_name());
	}

	nnet::mutable_connector<double>* mc =
			dynamic_cast<nnet::mutable_connector<double>*>(elem2);
	mc->add_arg(elem1, index);
}

}

#endif