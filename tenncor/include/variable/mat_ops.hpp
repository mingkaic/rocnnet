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

// TENSOR EXTENSION

template <typename T>
class extend : public iunar_ops<T> {
private:
	size_t index = 0;
	size_t multiplier = 0;
	WEAK_VAR_PTR<T> watch;

protected:
	virtual tensor<T>* calc_gradient (WEAK_VAR_PTR<T> over) const;
	virtual std::string get_symb (void) { return "extend"; }

	virtual void shape_eval (void);
	void copy (const ivariable<T>& other, std::string name = "");
	extend (const ivariable<T>& other, std::string name) { this->copy(other, name); }

	virtual std::shared_ptr<ivariable<T> > clone_impl (std::string name);

public:
	extend (void) {}
	extend (VAR_PTR<T> in);
	extend (VAR_PTR<T> in, WEAK_VAR_PTR<T> watch); // extend to fit shape
	extend (VAR_PTR<T> in, size_t index, size_t multiplier);
	virtual extend<T>& operator = (const ivariable<T>& other);

	std::shared_ptr<extend<T> > clone (std::string name = "") {
		return std::static_pointer_cast<extend<T>, ivariable<T> >(clone_impl(name));
	}

	// set data
	void set_ext_info (WEAK_VAR_PTR<T> watch);
	void set_ext_info (size_t index, size_t multiplier);

	virtual const tensor<T>& eval (void);
};

// TENSOR COMPRESSION ALONG ONE DIMENSION

template <typename T>
class compress : public iunar_ops<T> {
private:
	signed index = -1; // negative index indicates compression across all dimension
	// first parameter is the collecting buffer, second is the gathered data
	std::function<T(const std::vector<T>&)> collector; // default to average sum

protected:
	virtual tensor<T>* calc_gradient (WEAK_VAR_PTR<T> over) const;
	virtual std::string get_symb (void) { return "compress"; }

	virtual void shape_eval (void);
	void copy (const ivariable<T>& other, std::string name = "");
	compress (const ivariable<T>& other, std::string name) { this->copy(other, name); }

	virtual std::shared_ptr<ivariable<T> > clone_impl (std::string name);

public:
	compress (void) {}
	compress (VAR_PTR<T> in);
	compress (VAR_PTR<T> in, size_t index);
	compress (VAR_PTR<T> in, size_t index, std::function<T(const std::vector<T>&)> collector);
	virtual compress<T>& operator = (const ivariable<T>& other);

	std::shared_ptr<compress<T> > clone (std::string name = "") {
		return std::static_pointer_cast<compress<T>, ivariable<T> >(clone_impl(name));
	}

	void set_cmpr_info (size_t index, std::function<T(const std::vector<T>&)> collector);

	virtual const tensor<T>& eval (void);
};

// MATRIX TRANSPOSE

template <typename T>
class transpose : public iunar_ops<T> {
protected:
	// backward chaining for AD
	virtual tensor<T>* calc_gradient (WEAK_VAR_PTR<T> over) const;
	virtual std::string get_symb (void) { return "transpose"; }

	virtual void shape_eval (void);
	transpose (const ivariable<T>& other, std::string name) { this->copy(other, name); }

	virtual std::shared_ptr<ivariable<T> > clone_impl (std::string name);

public:
	transpose (void) {}
	transpose (VAR_PTR<T> in);

	std::shared_ptr<transpose<T> > clone (std::string name = "") {
		return std::static_pointer_cast<transpose<T>, ivariable<T> >(clone_impl(name));
	}

	virtual const tensor<T>& eval (void);
};

// MATRIX MULTIPLICATION

// restricted to 2-d matrices with proper shapes
// dimension 1 denote number of columns,
// dimension 2 denote number of rows
template <typename T>
class matmul : public ioperation<T> {
private:
	VAR_PTR<T> a = nullptr;
	VAR_PTR<T> b = nullptr;
	bool transposeA;
	bool transposeB;

protected:
	// backward chaining for AD
	virtual tensor<T>* calc_gradient (WEAK_VAR_PTR<T> over) const;
	virtual void replace (ivariable<T>* food, VAR_PTR<T> newfood) {
		if (a.get() == food) a = newfood;
		if (b.get() == food) b = newfood;
	}

	virtual void shape_eval (void);
	matmul (const matmul<T>& other, std::string name);

	virtual std::shared_ptr<ivariable<T> > clone_impl (std::string name);

public:
	matmul (void) : transposeA(false), transposeB(false) {}
	matmul (VAR_PTR<T> a, VAR_PTR<T> b,
			bool transposeA = false, bool transposeB = false);
	virtual ivariable<T>& operator () (VAR_PTR<T> a, VAR_PTR<T> b,
			bool transposeA = false, bool transposeB = false);
	virtual matmul<T>& operator = (const ivariable<T>& other);

	std::shared_ptr<matmul<T> > clone (std::string name = "") {
		return std::static_pointer_cast<matmul<T>, ivariable<T> >(clone_impl(name));
	}

	virtual const tensor<T>& eval (void);
};

}

#include "../../src/variable/mat_ops.tpp"

#endif /* matop_hpp */
