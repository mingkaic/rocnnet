//
//  functions.tpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef functions_ops

namespace nnet {

// FUNCTION WRAPPER IMPLEMENTATION

template <typename T>
void univar_func<T>::clear (void) {
    for (ioperation<T>* ptrs : ownout) {
        delete ptrs;
    }
    fanin = fanout = nullptr;
}

template <typename T>
void univar_func<T>::copy (const ivariable<T>& other, std::string name) {
    if (const univar_func<T>* uptr = dynamic_cast<const univar_func<T>*>(&other)) {
    	// deep copy:
    	std::queue<ivariable<T>*> q;
    	std::unordered_map<ivariable<T>*, ivariable<T>*> src_to_dest;
		ivariable<T>* buffer = nullptr;
		ivariable<T>* current = uptr->fanout->clone();
    	q.push(fanout);
    	src_to_dest[uptr->fanout] = current;

    	while (false == q.empty()) {
			buffer = q.front();
			current = src_to_dest[buffer];
			q.pop();
			if (iunar_ops<T>* op = dynamic_cast<iunar_ops<T>*>(buffer)) {
				iunar_ops<T>* cur = dynamic_cast<iunar_ops<T>*>(current);
				ivariable<T>* cpy = nullptr;
				if (src_to_dest.end() == src_to_dest.find(op->var)) {
					cpy = op->var->clone();
					q.push(op->var);
					src_to_dest[op->var] = cpy;
				} else {
					cpy = src_to_dest[op->var];
				}
				(*cur)(*cpy);
			} else if (ibin_ops<T>* op = dynamic_cast<ibin_ops<T>*>(buffer)) {
				ibin_ops<T>* cur = dynamic_cast<ibin_ops<T>*>(current);
				ivariable<T>* acpy = nullptr;
				ivariable<T>* bcpy = nullptr;
				if (src_to_dest.end() == src_to_dest.find(op->a)) {
					acpy = op->a->clone();
					q.push(op->a);
					src_to_dest[op->a] = acpy;
				} else {
					acpy = src_to_dest[op->a];
				}
				if (src_to_dest.end() == src_to_dest.find(op->b)) {
					bcpy = op->b->clone();
					q.push(op->b);
					src_to_dest[op->b] = bcpy;
				} else {
					bcpy = src_to_dest[op->b];
				}
				(*cur)(*acpy, *bcpy);
			} else {
				fanin = src_to_dest[buffer];
			}
    	}
    	this->consume(*fanin);
//        // TODO: make deep copy
//        // currently shallow copy
//        // no ownership
//        fanout = uptr->fanout;
//        fanin = uptr->fanin;
    }
    ivariable<T>::copy(other, name);
}

template <typename T>
univar_func<T>::univar_func (const univar_func& other, std::string name) {
    copy(other, name);
}

template <typename T>
void univar_func<T>::shape_eval (void) {
    tensor_shape ts = fanout->get_shape();
    if (ts.is_fully_defined()) {
        this->update(ts);
    }
}

template <typename T>
univar_func<T>::univar_func (
    std::function<void(ioperation<T>*&)> declare) {
    declare(fanout);
}

template <typename T>
univar_func<T>* univar_func<T>::clone (std::string name) {
    return new univar_func<T>(*this, name);
}

template <typename T>
ivariable<T>& univar_func<T>::operator () (ivariable<T>& input) {
	if (fanin) {
		std::unordered_set<ioperation<T>*> copy = fanin->get_consumers();
		for (ioperation<T>* cons : copy) {
			cons->deconsume(*fanin);
		}
	}
	ioperation<T>* buffer;
	std::queue<ioperation<T>*> q;
	q.push(fanout);
	// connect input
	while (false == q.empty()) {
		buffer = q.front();
		q.pop();
		if (iunar_ops<T>* uptr = dynamic_cast<iunar_ops<T>*>(buffer)) {
			if (nullptr == uptr->var) {
				(*uptr)(input);
			} else if (ioperation<T>* inptr =
				dynamic_cast<ioperation<T>*>(uptr->var)) {
				q.push(inptr);
			}
		} else if (ibin_ops<T>* bptr = dynamic_cast<ibin_ops<T>*>(buffer)) {
			if (nullptr == bptr->a && nullptr == bptr->b) {
				(*bptr)(input, input);
			} else {
				if (ioperation<T>* ptr1 =
					dynamic_cast<ioperation<T>*>(bptr->a)) {
					q.push(ptr1);
				}
				if (ioperation<T>* ptr2 =
					dynamic_cast<ioperation<T>*>(bptr->b)) {
					q.push(ptr2);
				}
			}
		}
	}
	fanin = &input;
	this->consume(input);
	if (session::pre_shape_eval()) {
		shape_eval();
	}
    return *this;
}

template <typename T>
univar_func<T>& univar_func<T>::operator = (const ivariable<T>& other) {
    if (this != &other) {
        clear();
        copy(other);
    }
    return *this;
}

template <typename T>
tensor<T>* univar_func<T>::gradient (ivariable<T>* over) const {
    return fanout->gradient(over);
}

template <typename T>
tensor<T>* univar_func<T>::gradient (void) const {
    return fanout->gradient(fanin);
}

template <typename T>
const tensor<T>& univar_func<T>::eval (void) {
    return fanout->eval();
}

// ACTIVATION FUNCTION

template <typename T>
sigmoid<T>::sigmoid (void) : univar_func<T>([this](ioperation<T>*& outop) {
    // f(x) = 1/(1+e^-x)
    ioperation<T>* negres = new neg<T>();
    ioperation<T>* expres = new exp<T>(*negres);
    ioperation<T>* denom = new add<T>(1, *expres);
    outop = new div<T>(1, *denom);
    this->ownout = { negres, expres, denom, outop };
}) {
	this->name = "sigmoid";
}

template <typename T>
tanh<T>::tanh (void) : univar_func<T>([this](ioperation<T>*& outop) {
    // f(x) = (e^(2*x)+1)/(e^(2*x)-1)
    ioperation<T>* pres = new add<T>(); // 2*x
    ioperation<T>* expres = new exp<T>(*pres);
    ioperation<T>* numer = new sub<T>(*expres, 1);
    ioperation<T>* denom = new add<T>(*expres, 1);
    outop = new div<T>(*numer, *denom);
    this->ownout = { pres, expres, numer, denom, outop };
}) {
	this->name = "tanh";
}

}

#endif
