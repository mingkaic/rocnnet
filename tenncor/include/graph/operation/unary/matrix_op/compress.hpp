//
//  compress.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifndef compress_hpp
#define compress_hpp

#include "graph/operation/unary/iunar_ops.hpp"

namespace nnet {

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
			VAR_PTR<T> root = ivariable<T>::make_shared(new compress(in, index));
			// TODO: come up with a dryer solution to handling inherited attribute nodes (perhaps treat every node as inherited?)
			// have each argument evaluate interaction root
			in->interact(root);
			return
		}
		static VAR_PTR<T> make (VAR_PTR<T> in, size_t index, std::function<T(const std::vector<T>&)> collector) {
			VAR_PTR<T> root = ivariable<T>::make_shared(new compress(in, index, collector));
			// TODO: come up with a dryer solution to handling inherited attribute nodes (perhaps treat every node as inherited?)
			// have each argument evaluate interaction root
			in->interact(root);
			return root;
		}
		virtual compress<T>& operator = (const ivariable<T>& other);

		std::shared_ptr<compress<T> > clone (std::string name = "") {
			return std::static_pointer_cast<compress<T>, ievoker<T> >(clone_impl(name));
		}

		void set_cmpr_info (size_t index, std::function<T(const std::vector<T>&)> collector);

		virtual const tensor<T>& eval (void);
};

}

#include "../../../../../src/graph/operation/unary/matrix_op/compress.ipp"

#endif /* compress_hpp */
