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
#include "unar_ops.hpp"

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
		virtual void make_gradient (VAR_PTR<T>& safety_ref);
		virtual std::string get_symb (void) { return "extend"; }

		virtual void shape_eval (void);
		void copy (const ivariable<T>& other, std::string name = "");
		extend (const ivariable<T>& other, std::string name) { this->copy(other, name); }
		extend (VAR_PTR<T> in, WEAK_VAR_PTR<T> watch); // extend to fit shape
		extend (VAR_PTR<T> in, size_t index, size_t multiplier);

		virtual EVOKER_PTR<T> clone_impl (std::string name);

	public:
		static VAR_PTR<T> make (VAR_PTR<T> in, WEAK_VAR_PTR<T> watch) {
			return ivariable<T>::make_shared(new extend(in, watch));
		}
		static VAR_PTR<T> make (VAR_PTR<T> in, size_t index = 0, size_t multiplier = 1) {
			return ivariable<T>::make_shared(new extend(in, index, multiplier));
		}
		virtual extend<T>& operator = (const ivariable<T>& other);

		std::shared_ptr<extend<T> > clone (std::string name = "") {
			return std::static_pointer_cast<extend<T>, ievoker<T> >(clone_impl(name));
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
		virtual void make_gradient (VAR_PTR<T>& safety_ref);
		virtual std::string get_symb (void) { return "compress"; }

		virtual void shape_eval (void);
		void copy (const ivariable<T>& other, std::string name = "");
		compress (const ivariable<T>& other, std::string name) { this->copy(other, name); }
		compress (VAR_PTR<T> in, size_t index);
		compress (VAR_PTR<T> in, size_t index, std::function<T(const std::vector<T>&)> collector);

		virtual EVOKER_PTR<T> clone_impl (std::string name);

	public:
		static VAR_PTR<T> make (VAR_PTR<T> in, size_t index = 0) {
			return ivariable<T>::make_shared(new compress(in, index));
		}
		static VAR_PTR<T> make (VAR_PTR<T> in, size_t index, std::function<T(const std::vector<T>&)> collector) {
			return ivariable<T>::make_shared(new compress(in, index, collector));
		}
		virtual compress<T>& operator = (const ivariable<T>& other);

		std::shared_ptr<compress<T> > clone (std::string name = "") {
			return std::static_pointer_cast<compress<T>, ievoker<T> >(clone_impl(name));
		}

		void set_cmpr_info (size_t index, std::function<T(const std::vector<T>&)> collector);

		virtual const tensor<T>& eval (void);
};

// MATRIX TRANSPOSE

template <typename T>
class transpose : public iunar_ops<T> {
	protected:
		// backward chaining for AD
		virtual void make_gradient (VAR_PTR<T>& safety_ref);
		virtual std::string get_symb (void) { return "transpose"; }

		virtual void shape_eval (void);
		transpose (const ivariable<T>& other, std::string name) { this->copy(other, name); }
		transpose (VAR_PTR<T> in);

		virtual EVOKER_PTR<T> clone_impl (std::string name);

	public:
		static VAR_PTR<T> make (VAR_PTR<T> in) {
			return ivariable<T>::make_shared(new transpose(in));
		}

		std::shared_ptr<transpose<T> > clone (std::string name = "") {
			return std::static_pointer_cast<transpose<T>, ievoker<T> >(clone_impl(name));
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
		virtual void make_gradient (VAR_PTR<T>& safety_ref);
		virtual void replace (ivariable<T>* food, VAR_PTR<T> newfood) {
			if (a.get() == food) a = newfood;
			if (b.get() == food) b = newfood;
		}

		virtual void shape_eval (void);
		matmul (const matmul<T>& other, std::string name);
		matmul (VAR_PTR<T> a, VAR_PTR<T> b, bool transposeA, bool transposeB);

		virtual EVOKER_PTR<T> clone_impl (std::string name);

	public:
		static VAR_PTR<T> make (VAR_PTR<T> a, VAR_PTR<T> b, bool transposeA = false, bool transposeB = false) {
			return ivariable<T>::make_shared(new matmul(a, b, transposeA, transposeB));
		}

		virtual matmul<T>& operator = (const ivariable<T>& other);

		std::shared_ptr<matmul<T> > clone (std::string name = "") {
			return std::static_pointer_cast<matmul<T>, ievoker<T> >(clone_impl(name));
		}

		virtual const tensor<T>& eval (void);
};

}

#include "../../src/variable/mat_ops.tpp"

#endif /* matop_hpp */
