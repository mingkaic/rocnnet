//
//  mat_ops.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifndef matop_hpp
#define matop_hpp

#include "operation.hpp"

namespace nnet {

// MATRIX OPERATIONS

// UNARY MATRIX OPERATIONS

template <typename T>
class iunar_mat_ops : public ioperation<T> {
protected:
	ivariable<T>* var = nullptr;

	virtual void replace (
			const ivariable<T>& food,
			const ivariable<T>* newfood) {
		if (var == &food) var = const_cast<ivariable<T>*>(newfood);
	}
	virtual std::string get_symb (void) = 0;

	virtual void shape_eval (void) = 0;
	void copy (const ivariable<T>& other, std::string name = ""); // virtually identical to iunar_ops

public:
	virtual ~iunar_mat_ops (void) {
		if (var) var->get_consumers().erase(this);
	}
	virtual ivariable<T>& operator () (ivariable<T>& in); // identical to iunar_ops
	virtual iunar_mat_ops<T>& operator = (const ivariable<T>& other);

	// eval remains abstract
};

// TENSOR EXTENSION

template <typename T>
class extend : public iunar_mat_ops<T> {
private:
	size_t index = 0;
	size_t multiplier = 0;
	ivariable<T>* watch = nullptr;

protected:
	virtual tensor<T>* calc_derive (ivariable<T>* over) const;
	virtual std::string get_symb (void) { return "extend"; }

	virtual void shape_eval (void);
	void copy (const ivariable<T>& other, std::string name = "");
	extend (const ivariable<T>& other, std::string name) { this->copy(other, name); }

public:
	extend (void) {}
	extend (ivariable<T>& in);
	extend (ivariable<T>& in, ivariable<T>* watch); // extend to fit shape
	extend (ivariable<T>& in, size_t index, size_t multiplier);
	void set_ext_info (ivariable<T>* watch);
	void set_ext_info (size_t index, size_t multiplier);
	virtual extend<T>* clone (std::string name = "");
	virtual extend<T>& operator = (const ivariable<T>& other);

	virtual const tensor<T>& eval (void);
};

// TENSOR COMPRESSION ALONG ONE DIMENSION

template <typename T>
class compress : public iunar_mat_ops<T> {
private:
	signed index = -1; // negative index indicates compression across all dimension
	// first parameter is the collecting buffer, second is the gathered data
	std::function<T(const std::vector<T>&)> collector; // default to average sum

protected:
	virtual tensor<T>* calc_derive (ivariable<T>* over) const;
	virtual std::string get_symb (void) { return "compress"; }

	virtual void shape_eval (void);
	void copy (const ivariable<T>& other, std::string name = "");
	compress (const ivariable<T>& other, std::string name) { this->copy(other, name); }

public:
	compress (void) {}
	compress (ivariable<T>& in);
	compress (ivariable<T>& in, size_t index);
	compress (ivariable<T>& in, size_t index, std::function<T(const std::vector<T>&)> collector);
	virtual compress<T>* clone (std::string name = "");
	virtual compress<T>& operator = (const ivariable<T>& other);

	void set_cmpr_info (size_t index, std::function<T(const std::vector<T>&)> collector);

	virtual const tensor<T>& eval (void);
};

// MATRIX TRANSPOSE

template <typename T>
class transpose : public iunar_mat_ops<T> {
protected:
	// backward chaining for AD
	virtual tensor<T>* calc_derive (ivariable<T>* over) const;
	virtual std::string get_symb (void) { return "transpose"; }

	virtual void shape_eval (void);
	transpose (const ivariable<T>& other, std::string name) { this->copy(other, name); }

public:
	transpose (void) {}
	transpose (ivariable<T>& in);
	virtual transpose<T>* clone (std::string name = "");

	virtual const tensor<T>& eval (void);
};

// MATRIX MULTIPLICATION

// restricted to 2-d matrices with proper shapes
// dimension 1 denote number of columns,
// dimension 2 denote number of rows
template <typename T>
class matmul : public ioperation<T> {
private:
	ivariable<T>* a = nullptr;
	ivariable<T>* b = nullptr;
	bool transposeA;
	bool transposeB;

protected:
	// backward chaining for AD
	virtual tensor<T>* calc_derive (ivariable<T>* over) const;
	virtual void replace (
			const ivariable<T>& food,
			const ivariable<T>* newfood) {
		if (a == &food) a = const_cast<ivariable<T>*>(newfood);
		if (b == &food) b = const_cast<ivariable<T>*>(newfood);
	}

	virtual void shape_eval (void);
	matmul (const matmul<T>& other, std::string name);

public:
	matmul (void) : transposeA(false), transposeB(false) {}
	matmul (
			ivariable<T>& a,
			ivariable<T>& b,
			bool transposeA = false,
			bool transposeB = false);
	virtual matmul<T>* clone (std::string name = "");
	virtual ivariable<T>& operator () (
			ivariable<T>& a,
			ivariable<T>& b,
			bool transposeA = false,
			bool transposeB = false);
	virtual ~matmul (void) {
		if (a) a->get_consumers().erase(this);
		if (b) b->get_consumers().erase(this);
	}
	virtual matmul<T>& operator = (const ivariable<T>& other);

	virtual const tensor<T>& eval (void);
};

}

#include "../../src/variable/mat_ops.tpp"

#endif /* matop_hpp */
