#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" {
	/*DLLEXPORT int my_add(int x, int y) {
		return x + y;
	}

	DLLEXPORT int my_mul(int x, int y) {
		return x * y;
	}*/
	
	extern "C" {
    DLLEXPORT double* linear_model_create(int input_dim)
    {
        srand(time(NULL));
        double* tab = new double[input_dim + 1];
        for (int i = 0; i < input_dim + 1; i++)
        {
            tab[i] = (double)rand() / (double)RAND_MAX;
        }
        return tab;
    }

    DLLEXPORT double linear_model_predict_regression(double* model, double* inputs, int input_dim)
    {
        double product = 0;
        for (int i = 0; i < input_dim; ++i)
            product += model[i] * inputs[i];
        return product + model[0];
    }

    DLLEXPORT double linear_model_predict_classification(double* model, double* inputs, int input_dim)
    {
        return linear_model_predict_regression(model, inputs, input_dim) >= 0 ? 1 : -1;
    }

    DLLEXPORT double* linear_model_train_classification(
        double* model,
        double** dataset_inputs,
        int dataset_inputs_size,
        int input_dim,
        double* dataset_expected_outputs,
        int iterations_count = 100,
        double alpha = 0.01)
    {
        srand(time(NULL));
        for (int it = 0; it < iterations_count; it++)
        {
            int k = rand() % dataset_inputs_size;
            double g_x_k = linear_model_predict_classification(model, dataset_inputs[k], input_dim);
            double grad = alpha * (dataset_expected_outputs[k] - g_x_k);
            model[0] += grad * 1;
            for (int i = 0; i < input_dim; i++)
                model[i + 1] += grad * dataset_inputs[k][i];
        }
        return model;
    }
}
