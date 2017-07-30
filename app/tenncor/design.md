# Tenncor
### Summary
Tenncor differentiates values using Automatic Differentiation (AD) which maps each individual forward operation (addition, subtraction, multiplication, etc.) 
to some backward operation representing the forward's first derivative. An equation represented in this fashion can correspond to its gradient
by repeatedly applying chain rule to the backward operation of its forward operations.

### Design

Let us represent each operation as a type of node N where forward operations are N<sub>f</sub> and backward operations are N<sub>b</sub>

Let G represent a single equation instance; G is a graph of operation nodes N

Let F be a function mapping N<sub>f</sub> to N<sub>b</sub>; F<sub>G</sub>: N<sub>f</sub> &rightarrow; N<sub>b</sub>

F(N<sub>leaf</sub>) = {<br>
&nbsp;&nbsp;&nbsp;&nbsp; 1 if deriving wrt to N<sub>leaf</sub>, <br>
&nbsp;&nbsp;&nbsp;&nbsp; 0 otherwise<br>
}

F(N<sub>f</sub>) = {<br>
&nbsp;&nbsp;&nbsp;&nbsp; 1 if deriving wrt to N<sub>f</sub>, <br>
&nbsp;&nbsp;&nbsp;&nbsp; B(N<sub>f</sub>) otherwise<br>
} where B is a backward mapper unique to each typeof N<sub>f</sub>

Let's join B(N<sub>f</sub>), F(N<sub>leaf</sub>) and F(N<sub>f</sub>)

F(N) = {<br>
&nbsp;&nbsp;&nbsp;&nbsp; 1 if deriving wrt to N<br>
&nbsp;&nbsp;&nbsp;&nbsp; 0 if typeof N is leaf<br>
&nbsp;&nbsp;&nbsp;&nbsp; [∆arg1, ∆arg2] if typeof N == add or sub<br>
&nbsp;&nbsp;&nbsp;&nbsp; ...<br>
&nbsp;&nbsp;&nbsp;&nbsp; matmul if typeof N == matmul<br>
&nbsp;&nbsp;&nbsp;&nbsp; ...<br>
}

For specific operations like matmul, its backwards edge involve modifying the calling root, let's define these operations as R

To avoid recomputing gradients, let us cache the gradient values in the node structure

	struct Node {
		struct metadata {
			string id
			string label
			stringref graphid
		}
		tensor data
		Nf forwardfunction
		F backwardfunction
		R modifyroot
		map<Nleaf, Nb> cache
	}
	
Operation graphs are built by composite nodes

	// inheritance hierarchy
	Node -> [NLeaf, NConnnector]
	NLeaf -> [Constant, NVariable]
	NConnnector -> [Operations, ...]
	
Leaf nodes don't need a cache, manipulate data, or manage graph info. 

And leaves all share the the same forward and backward function.
For connectors, gradient values don't need to be cached if they have only one consumer, since the value will be recalculated for each leaf update.
Additionally, connectors don't necessary have an active data whereas leaves and constants always have allocated data

	struct Node {
		string id
		string label
		F backwardfunction = 0
		[True, False] data_available;
	}
	
	struct NLeaf : public Node {
		tensor data
	}
	
	struct Constant : public NLeaf {
		F backwardfunction = [0]
		True data_available;
	}
	
	struct NVariable : public NLeaf {
		F backwardfunction = [1 if derive wrt this, 0 otherwise]
	}
		
	struct NConnnector : public Node {
		tensorref data = null
		stringref graphid
		map<Nleaf, Nb> cache
		R modifyroot = 0
		Nf forwardfunction = 0
	}

### Layers

Three layers become immediately obvious:

- Graph manage the distribution of tensor.
	Data is transfered between from sources (leaves) to sinks (roots).

- Tensor manage shape (dimensionality) information.

- Raw manage the allocation, deallocation, and serialization of raw data.

### Main Objects

Graph Layer

	- iobserver and subject enforce reactive behavior
	
	- leaves are graph subjects
	
	- connectors are graph subject, observer composites
	
	- iexecutor are graph observers and prevents updates

Tensor Layer

	- tensor wraps shape and raw data
	
	- tensorshape represents shape information
	
	- tensor manipulator is a delegate for changing raw data

Memory Layer

	- iallocator allows custom allocators for security or analysis purposes
	
	- session marks which nodes, tensors, and raw data to serialize
