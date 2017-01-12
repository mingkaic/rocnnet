//
//  mlp.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef mlp_hpp
#define mlp_hpp

#include "graph/functions.hpp"
#include "layer.hpp"
#include "innet.hpp"

namespace nnet
{

#define IN_PAIR std::pair<size_t, nnet::VAR_FUNC<double> >
#define HID_PAIR std::pair<layer_perceptron*, nnet::VAR_FUNC<double> >

class ml_perceptron : public innet
{
	protected:
		std::vector<HID_PAIR> layers;
		
		void copy (const ml_perceptron& other, std::string scope);
		ml_perceptron (const ml_perceptron& other, std::string scope);
		virtual ml_perceptron* clone_impl (std::string scope)
		{
			return new ml_perceptron (*this, scope);
		}

	public:
		// trust that passed in operations are unconnected
		ml_perceptron (size_t n_input, std::vector<IN_PAIR> hiddens,
			std::string scope = "MLP");
		virtual ~ml_perceptron (void);
		
		// COPY
        virtual ml_perceptron* clone (std::string scope = "");
		ml_perceptron& operator = (const ml_perceptron& other);

        // MOVE
        
        // PLACEHOLDER CONNECTION
		// input are expected to have shape n_input by batch_size
		// outputs are expected to have shape output by batch_size
		nnet::varptr<double> operator () (placeholder<double>* input);
		std::vector<WB_PAIR> get_variables (void);
};

}

#endif /* mlp_hpp */
