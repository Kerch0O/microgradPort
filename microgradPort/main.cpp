#include "functions.h"



int main() {

	//Using example from Andrej's youtube video
	
		
	std::vector<std::vector<float>> xs = { {2.0f, 3.0f, -1.0f},
											{3.0f, -1.0f, 0.5f},
											{0.5f, 1.0f, 1.0f},
											{1.0f, 1.0f, -1.0f} };

	std::vector<float> ys = { 1.0f, -1.0f, -1.0f, 1.0f };
	


	
	std::vector<Value*> ypred;
	std::vector<float> end;
	int max = 10000;
	
	float iters = 10;
	//create iters many networks and find the lowest cost each can get to with a max of 10000 learning iterations
	//networks easily get stuck in local minimas when weights are initialised randomly
	//lowest cost I've seen is 0.0005
	//I think there is an issue with memory leaks even when the deconstructor for each network frees the heap memory
	//This is likely due to intermediate Values which were created along the way but somehow got disconnected(?)

	
	for (int j = 0; j < iters; j++) {
		MLP nn(3, std::vector<float>{4, 4, 1});

		int curr = 0;
		Value* cost = nullptr;
		for (int i = 0; i < 50; i++) {
			ypred.erase(ypred.begin(), ypred.end());

			for (auto& x : xs) {
				ypred.push_back(nn.forward(x)[0]);//Only one input for this case

			}
			cost = loss(ys, ypred);

			for (auto& x : nn.parameters()) {
				x->grad = 0.0f;
			}

			cost->backward();

			for (auto& p : nn.parameters()) {
				p->data += -0.05f * p->grad;
			}

			curr++;
			if (cost->data <= 1.0f && curr < max - 50)i = 0;
		}
		end.push_back(cost->data);
		std::cout << "Ended on " << cost->data << std::endl;
	}

	float lowest = end[0];

	for (int i = 1; i < end.size(); i++) {
		if (lowest >= end[i]) {
			lowest = end[i];
		}
	}

	std::cout << "Best cost out of " << iters << " is: " << lowest << std::endl;




	return 0;
}