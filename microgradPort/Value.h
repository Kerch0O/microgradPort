#pragma once
#include "functions.h"


class Value {
public:

	float data, grad = 0.0f;
	std::pair<Value*, Value*> prev;

	Value* out;

	bool a_backward = false;

	char type = '1';

	Value(std::pair<Value*, Value*> children, const float& d);
	void _backward();


	void backward();
};
