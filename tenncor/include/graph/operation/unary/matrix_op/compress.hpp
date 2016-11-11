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

		compress (const compress<T>& other, std::string name) { copy(other, name); }

	protected:
		virtual void setup_gradient (void);
		virtual ievoker<T>* clone_impl (std::string name);
		virtual std::string get_symb (void) { return "compress"; }
		virtual void shape_eval (void);
		
		void copy (const ivariable<T>& other, std::string name = "");

	public:
		compress (ivariable<T>* in, size_t index);
		compress (ivariable<T>* in, size_t index, std::function<T(const std::vector<T>&)> collector);

		// COPY
        compress<T>* clone (std::string name = "") {
			return static_cast<compress<T>*>(clone_impl(name));
		}
		virtual compress<T>& operator = (const ivariable<T>& other);
		
		virtual void update (void);

		void set_cmpr_info (size_t index, std::function<T(const std::vector<T>&)> collector);
};

}

#include "../../../../../src/graph/operation/unary/matrix_op/compress.ipp"

#endif /* compress_hpp */
