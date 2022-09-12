#include "functions.h"



int main() {

	MLP nn(3, std::vector<float>{4, 4, 1});
	
	std::vector<std::vector<float>> xs = { {2.0f, 3.0f, -1.0f},
											{3.0f, -1.0f, 0.5f},
											{0.5f, 1.0f, 1.0f},
											{1.0f, 1.0f, -1.0f} };

	std::vector<float> ys = { 1.0f, -1.0f, -1.0f, 1.0f };
	std::vector<Value*> ypred;
	for (auto& x : xs) {
		ypred.push_back(nn.forward(x)[0]);//Only one input for this case
	}

	for (auto& x : ypred) {
		std::cout << x->data << std::endl;
	}

	Value* cost = loss(ys, ypred);
	std::cout << "Cost: " << cost->data << std::endl;

	cost->backward();
	std::cout << nn.layers[0].neurons[0].w[0]->grad << std::endl;

	for (int i = 0; i < 100; i++) {

		for (auto& p : nn.parameters()) {
			p->data += -0.01f * p->grad;
		}

		std::vector<Value*> npred;
		for (auto& x : xs) {
			npred.push_back(nn.forward(x)[0]);//Only one input for this case
		}

		//Do this stuff next i dont want Values to be written without updating because memory and efficiency and deviation from code
	}

	return 0;
}