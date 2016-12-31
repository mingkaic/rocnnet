//
// Created by Mingkai Chen on 2016-12-27.
//

#include <queue>
#include "graph/mutable/mutable_connector.hpp"
#include "graph/operation/elementary.hpp"
#include "graph/operation/transform.hpp"
#include "graph/operation/matmul.hpp"
#include "creator_vertex.hpp"

#ifdef creator_vertex_hpp

namespace tensorio
{

// adhoc function mapping connector types to connector constructors
static nnet::mutable_connector<double>* mutable_build (CONNECTOR_TYPE cm)
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
				return nnet::matmul<double>::build(args[0], args[1]);
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

struct vertex_manager::node_registry
{
	// pairs metainfo and ivariable pointer
	struct info_wrapper
	{
		nnet::ivariable<double>* ptr_;
		metainfo info_;
	};

	std::unordered_map<std::string,info_wrapper> nodes_;

	// delete all ptr_s in node_ registry
	~node_registry (void)
	{
		std::unordered_set<nnet::ivariable<double>*> indeps;
		for (auto ns : nodes_)
		{
			// only mark independent nodes and leaves for deletion
			if (ns.second.info_.op_type_)
			{
				nnet::mutable_connector<double>* mc = dynamic_cast<nnet::mutable_connector<double>*>(ns.second.ptr_);
				// invalid arguments means mutable connector is not dependent (mark to destroy)
				if (false == mc->valid_args())
				{
					indeps.emplace(ns.second.ptr_);
				}
			}
			// ns.second.ptr_ is a leaf (mark to destroy)
			else
			{
				indeps.emplace(ns.second.ptr_);
			}
		}
		// destroy
		for (nnet::ivariable<double>* v : indeps)
		{
			delete v;
		}
	}

	// check if id exists, extract the node, then return its type
	bool grab_node (info_wrapper& out, std::string id)
	{
		auto fit = nodes_.find(id);
		// return if id doesn't exist in repository
		if (nodes_.end() == fit)
		{
			return false;
		}
		out = fit->second;
		return true;
	}
};

// grab all connections under sub-network flow connected to root,
// as long as it's in unvisited,
// remove visited nodes from unvisited (BFS)
static void extract_connections (CONNECTION_SET& conns,
	nnet::iconnector<double>* root,
	std::function<bool(std::string)> unique_checking)  // check and update id
{
	// queue for breadth wise search
	std::queue<nnet::iconnector<double>*> mcq;
	// argument vector
	std::vector<nnet::ivariable<double>*> args;

	nnet::iconnector<double>* conn = root;
	std::string conn_id;
	mcq.push(conn);
	while (false == mcq.empty())
	{
		conn = mcq.front();
		mcq.pop();
		conn_id = conn->get_uid();
		if (unique_checking(conn_id))
		{
			conn->get_args(args);
			for (size_t i = 0; i < args.size(); i++)
			{
				// only bother with connectors
				if (nnet::iconnector<double>* ac =
					dynamic_cast<nnet::iconnector<double>*>(args[i]))
				{
					mcq.push(ac);
				}

				if (nullptr != args[i])
				{
					connection c;
					c.from_id = args[i]->get_uid();
					c.to_id = conn_id;
					c.idx = i;
					conns.emplace(c);
				}
			}
		}
	}
}

vertex_manager::vertex_manager (void) : inst(new node_registry()) {}

vertex_manager::~vertex_manager (void) { delete inst; }

std::string vertex_manager::register_op (CONNECTOR_TYPE cm)
{
	node_registry::info_wrapper wrap;
	nnet::mutable_connector<double>* con = mutable_build(cm);
	std::string id = con->get_uid(); // get uid
	wrap.ptr_ = con;
	wrap.info_.op_type_ = cm; // store connector type
	inst->nodes_[id] = wrap;
	return id;
}

std::string vertex_manager::register_leaf (std::string label, var_opt opt)
{
	nnet::ileaf<double>* leaf = nullptr;
	switch (opt.type)
	{
		case PLACE:
			leaf = new nnet::placeholder<double>(opt.shape_, label);
			break;
		case CONST:
			{
				nnet::const_init<double> init(opt.parameter_->val_);
				leaf = new nnet::variable<double>(opt.shape_, init, label);
			}
			break;
		case RAND:
			{
				double min = opt.parameter_->min2max_.first;
				double max = opt.parameter_->min2max_.second;
				nnet::random_uniform<double> init(min, max);
				leaf = new nnet::variable<double>(opt.shape_, init, label);
			}
			break;
		default:
			// error
			throw std::invalid_argument("poorly defined variable option");
	}
	std::string id = leaf->get_uid();
	// store data in registry
	node_registry::info_wrapper wrap;
	wrap.ptr_ = leaf;
	inst->nodes_[id] = wrap;
	return id;
}

bool vertex_manager::delete_node (std::string id)
{
	node_registry::info_wrapper wrapper;
	if (inst->grab_node(wrapper, id))
	{
		inst->nodes_.erase(id);
		// unattach permanent node from mutable_connector wrapper
		ccoms::update_message msg;
		msg.cmd_ = ccoms::update_message::REMOVE_ARG;
		wrapper.ptr_->notify(msg);
		// delete wrapper.ptr_: triggering chain destruction
		delete wrapper.ptr_;
		return true;
	}
	return false;
}

void vertex_manager::link (std::string id1, std::string id2, size_t index)
{
	node_registry::info_wrapper w1;
	node_registry::info_wrapper w2;

	bool g1 = inst->grab_node(w1, id1);
	// error: one non-existent node
	if (false == g1 ||
		false == inst->grab_node(w2, id2))
	{
		std::string invalid_id = g1 ? id1 : id2;
		throw std::invalid_argument("invalid id: (" + invalid_id + ") node not found");
	}

	// error: can't connect to a leaf
	if (!w2.info_.op_type_)
	{
		throw std::invalid_argument("can't connect to leaf " + w2.ptr_->get_name());
	}

	nnet::mutable_connector<double>* mc =
			static_cast<nnet::mutable_connector<double>*>(w2.ptr_);
	mc->add_arg(w1.ptr_, index);
}

bool vertex_manager::delete_link (std::string id, size_t index)
{
	node_registry::info_wrapper wrap;
	// error: one non-existent node
	if (false == inst->grab_node(wrap, id) || !wrap.info_.op_type_)
	{
		return false;
	}

	static_cast<nnet::mutable_connector<double>*>(wrap.ptr_)->remove_arg(index);
	return true;
}

std::experimental::optional<metainfo> vertex_manager::node_info (std::string id) const
{
	std::experimental::optional<metainfo> info;
	node_registry::info_wrapper wrap;
	if (inst->grab_node(wrap, id))
	{
		info = wrap.info_;
	}
	return info;
}

void vertex_manager::get_connections (CONNECTION_SET& conns, std::string root_id) const
{
	conns.clear();
	// uniquely store to nodes to prevent from copying duplicates
	std::unordered_set<std::string> uniqueness;
	node_registry::info_wrapper wrap;
	if (inst->grab_node(wrap, root_id))
	{
		if (nnet::iconnector<double>* con = dynamic_cast<nnet::iconnector<double>*>(wrap.ptr_))
		{
			extract_connections(conns, con,
			[&uniqueness](std::string id)
			{
				if (uniqueness.end() == uniqueness.find(id))
				{
					uniqueness.emplace(id);
					return true;
				}
				return false;
			});
		}
	}
}

void vertex_manager::get_forwards (std::unordered_set<std::string>& ids, CONNECTION_SET& conns) const
{
	ids.clear();
	conns.clear();
	for (auto ns : inst->nodes_)
	{
		ids.emplace(ns.first);
	}

	// grab connections...
	// by traversing down a selected root, marking all visited nodes, and avoid visited nodes
	// keep selecting roots (that are not visited) until all nodes are visited
	std::unordered_set<std::string> unvisited = ids; // copy
	std::string cid;
	while (false == unvisited.empty())
	{
		cid = *(unvisited.begin());
		node_registry::info_wrapper wrap;
		if (inst->grab_node(wrap, cid))
		{
			if (nnet::iconnector<double>* con = dynamic_cast<nnet::iconnector<double>*>(wrap.ptr_))
			{
				extract_connections(conns, con,
				[&unvisited](std::string id)
				{
					if (unvisited.end() != unvisited.find(id))
					{
						unvisited.erase(id);
						return true;
					}
					return false;
				});
			}
		}
		unvisited.erase(cid); // remove cid no matter what
	}
}

void vertex_manager::get_backwards (std::unordered_set<std::string>& ids, CONNECTION_SET& conns) const
{
	ids.clear();
	conns.clear();
	std::unordered_set<nnet::iconnector<double>*> sub_roots;
	nnet::iconnector<double>* grad;
	for (auto ns : inst->nodes_)
	{
		grad = ns.second.ptr_->get_gradient();
		ids.emplace(grad->get_uid());
		sub_roots.emplace(grad);
	}
	// similar to get_forward, we grab connections of selected roots of sub-trees
	// but since grads are roots mini operation trees it can have
	// multiple non-subroot nodes that we should include;
	// we include any node that is not a sub_root (in sr_ids)
	std::unordered_set<std::string> unvisited = ids;
	std::unordered_set<std::string> nonsr_id; // store non-subroots
	std::string cid;
	while (false == unvisited.empty())
	{
		cid = *(unvisited.begin());
		node_registry::info_wrapper wrap;
		if (inst->grab_node(wrap, cid))
		{
			if (nnet::iconnector<double>* con = dynamic_cast<nnet::iconnector<double>*>(wrap.ptr_))
			{
				extract_connections(conns, con,
				[ids, &unvisited, &nonsr_id](std::string id)
				{
					if (unvisited.end() != unvisited.find(id))
					{
						unvisited.erase(id);
						return true;
					}
					// id is not a gradient sub root
					else if (ids.end() == ids.find(id))
					{
						nonsr_id.emplace(id);
						return true; // continue down the tree if possible
					}
					return false;
				});
			}
		}
		unvisited.erase(cid); // remove cid no matter what
	}
}

}

#endif