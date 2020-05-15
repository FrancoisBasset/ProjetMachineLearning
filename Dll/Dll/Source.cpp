#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

using Eigen::MatrixXd;

struct MLP {
    int* npl;
    int npl_size;
    double*** w;
    double** x;
    double** deltas;
};

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
            product += model[i + 1] * inputs[i];
        return product + model[0];
    }

    DLLEXPORT double linear_model_predict_classification(double* model, double* inputs, int input_dim)
    {
        return linear_model_predict_regression(model, inputs, input_dim) >= 0 ? 1 : -1;
    }

    DLLEXPORT void linear_model_train_classification(
        double* model,
        double* dataset_inputs,
        int dataset_inputs_line_count,
        int input_dim,
        double* dataset_expected_outputs,
        int iterations_count = 1000000,
        double alpha = 0.001)
    {
        srand(time(NULL));
        for (int it = 0; it < iterations_count; it++)
        {
            int k = rand() % dataset_inputs_line_count;
            double* dataset_inputs_k = new double[input_dim];
            for (int i = 0; i < input_dim; i++)
                dataset_inputs_k[i] = dataset_inputs[k * input_dim + i];
            double g_x_k = linear_model_predict_classification(model, dataset_inputs_k, input_dim);
            double grad = alpha * (dataset_expected_outputs[k] - g_x_k);
            model[0] += grad * 1;
            for (int i = 0; i < input_dim; i++)
                model[i + 1] += grad * dataset_inputs[k * input_dim + i];
        }
    }
		
    DLLEXPORT void linear_model_train_regression(double* model, double* dataset_inputs, int dataset_inputs_line_count, int input_dim, double* dataset_expected_outputs)
    {
        // transposée de X
        double** XT = new double* [input_dim + 1];
        for (int i = 0; i < input_dim + 1; i++)
        {
            XT[i] = new double[dataset_inputs_line_count];
            for (int j = 0; j < dataset_inputs_line_count; j++)
                XT[i][j] = dataset_inputs[j + i * 2];
        }
        // XT * X
        int i, j, k;
        double** product = new double* [input_dim + 1];
        for (i = 0; i < input_dim + 1; i++)
        {
            product[i] = new double[input_dim + 1];
            for (j = 0; j < input_dim + 1; j++)
            {
                product[i][j] = 0;
                for (k = 0; k < dataset_inputs_line_count; k++)
                    product[i][j] += XT[i][k] * dataset_inputs[k * (input_dim + 1) + j];
            }
        }
        // inverse
        MatrixXd m(input_dim + 1, input_dim + 1);
        for (i = 0; i < input_dim + 1; i++)
            for (j = 0; j < input_dim + 1; j++)
                m(i, j) = product[i][j];
        MatrixXd inv = m.inverse();
        // inverse * transposée
        MatrixXd mXT(input_dim + 1, dataset_inputs_line_count);
        for (i = 0; i < input_dim + 1; i++)
            for (j = 0; j < dataset_inputs_line_count; j++)
                mXT(i, j) = XT[i][j];
        MatrixXd XTxInv = inv * mXT;
        // * Y
        MatrixXd mY(dataset_inputs_line_count, 1);
        for (i = 0; i < dataset_inputs_line_count; i++)
            mY(i, 0) = dataset_expected_outputs[i];
        MatrixXd mRes = XTxInv * mY;
        for (i = 0; i < input_dim + 1; i++) {
            std::cout << model[i] << std::endl;
            model[i] = mRes(i, 0);
        }
    }

    DLLEXPORT MLP* mlp_create(int* npl, int npl_size)
    {
        auto model = new MLP();
        model->npl = new int[npl_size];
        for (int i = 0; i < npl_size; i++)
            model->npl[i] = npl[i];
        model->npl_size = npl_size;
        model->w = new double** [npl_size];
        for (int l = 1; l < npl_size; l++)
        {
            model->w[l] = new double* [npl[l - 1] + 1];
            for (int i = 0; i < npl[l - 1] + 1; i++)
                model->w[l][i] = new double[npl_size];
        }
        model->x = new double* [npl_size];
        model->deltas = new double* [npl_size];
        for (int l = 0; l < npl_size; l++)
        {
            model->x[l] = new double[npl[l] + 1];
            model->x[l][0] = 1.0;
            model->deltas[l] = new double[npl[l] + 1];
        }
        return model;
    }
		
    DLLEXPORT void mlp_propagation(MLP* model, double* inputs, bool regression)
    {
        for (int j = 1; j < model->npl[0] + 1; j++)
            model->x[0][j] = inputs[j - 1];
        for (int l = 1; l < model->npl_size; l++)
        {
            for (int j = 1; j < model->npl[l] + 1; j++)
            {
                auto sum = 0.0;
                for (auto i = 0; i < model->npl[l - 1] + 1; i++)
                    sum += model->w[l][i][j] * model->x[l - 1][i];
                model->x[l][j] = (l == model->npl_size - 1 && regression) ? sum : tanh(sum);
            }
        }
    }
		
    DLLEXPORT double* mlp_propagation_and_extract_result(MLP* model, double* inputs, bool regression)
    {
        mlp_propagation(model, inputs, regression);
        auto rslt = new double[model->npl[model->npl_size - 1]];
        for (int j = 1; j < model->npl[model->npl_size - 1] + 1; j++)
        {
            rslt[j - 1] = model->x[model->npl_size - 1][j];
        }
        return rslt;
    }
		
    DLLEXPORT double* mlp_model_predict_regression(MLP* model, double* inputs)
    {
        return mlp_propagation_and_extract_result(model, inputs, true);
    }

    DLLEXPORT double* mlp_model_predict_classification(MLP* model, double* inputs)
    {
        return mlp_propagation_and_extract_result(model, inputs, false);
    }
		
    DLLEXPORT void mlp_model_train_classification(struct MLP* model, double* dataset_inputs, int dataset_length, int inputs_size,
        double* dataset_expected_outputs, int outputs_size, int iterations_count, double alpha)
    {
        for (auto it = 0; it < iterations_count; it++)
        {
            auto k = (int)floor(((double)std::min(rand(), RAND_MAX - 1) / RAND_MAX * dataset_length));
            auto inputs = dataset_inputs + k * inputs_size;
            auto expected_outputs = dataset_expected_outputs + k * outputs_size;
            auto rslt = mlp_propagation_and_extract_result(model, inputs, false);
            delete[] rslt;
            for (auto j = 1; j < model->npl[model->npl_size - 1] + 1; j++)
            {
                model->deltas[model->npl_size - 1][j] = (1 - pow(model->x[model->npl_size - 1][j], 2)) *
                    (model->x[model->npl_size - 1][j] - expected_outputs[j - 1]);
            }
            for (auto l = model->npl_size - 1; l >= 2; l--)
            {
                for (auto i = 1; i < model->npl[l - 1] + 1; i++)
                {
                    auto sum = 0.0;
                    for (auto j = 1; j < model->npl[l] + 1; j++)
                    {
                        sum += model->w[l][i][j] * model->deltas[l][j];
                    }
                    model->deltas[l - 1][i] = (1 - pow(model->x[l - 1][i], 2)) * sum;
                }
            }
            for (auto l = 1; l < model->npl_size; l++)
            {
                for (auto i = 1; l < model->npl[l - 1]; i++)
                {
                    for (auto j = 1; l < model->npl[l]; j++)
                    {
                        model->w[l][i][j] -= alpha * model->x[l - 1][i] * model->deltas[l][j];
                    }
                }
            }
        }
    }
}
