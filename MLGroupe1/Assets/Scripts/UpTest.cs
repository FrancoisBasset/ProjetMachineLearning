#define OK

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using System.Runtime.InteropServices;

public class UpTest : MonoBehaviour
{
    // Start is called before the first frame update
    public Transform[] trainSpheresTransforms;

    public Transform[] testSpheresTransforms;

   

    public void TrainAndTest()
    {
        //test = rouge
        //train = blanche
        int input_dim = trainSpheresTransforms.Length * 2;

        double[] input_tabs = new double[input_dim];
        int cpt = 0;
        foreach (Transform item in trainSpheresTransforms)
        {
            

            double y = item.position.y;
            double z = item.position.z;

            input_tabs[cpt] = y;
            input_tabs[cpt+1] = z;
            //input_tabs[cpt+2] = z;

            cpt += 2;
        }
        IntPtr model = MLWrapper.linear_model_create(input_dim);
                int input_pointer_size = Marshal.SizeOf(input_tabs[0]) * input_tabs.Length;

        IntPtr input_pointer = Marshal.AllocHGlobal(input_pointer_size);



        try
        {
            UnityEngine.Debug.Log($"Debut  = {input_tabs.Length } // {input_dim * Marshal.SizeOf(1d)}");

            Marshal.Copy(input_tabs, 0, input_pointer, input_tabs.Length);
            UnityEngine.Debug.Log($"model  = {model}");
            float p = MLWrapper.linear_model_predict_classification(model, input_pointer, input_dim);
            UnityEngine.Debug.Log($"predict = {p}");


            MLWrapper.linear_model_train_classification(model,input_pointer,0,input_dim,IntPtr.Zero,10000,0.01);

            float res = MLWrapper.linear_model_predict_classification(model, input_pointer, input_dim);
            UnityEngine.Debug.Log($"res  = {res} // {input_tabs.Length}");
        }
        catch (Exception e)
        {

            UnityEngine.Debug.Log($"e  = {e.Message} // lenght  = {input_tabs.Length}");

        }
        finally
        {
            Marshal.FreeHGlobal(input_pointer);
        }






    }

}

public static class MLWrapper
{

    private const string DLL = "2020_5A_AL2_Library_VisualStudio_Cpp";

#if (OK)
    
    [DllImport(DLL)]
    public static extern IntPtr linear_model_create(int input_dim);

    [DllImport(DLL)]
    public static extern float linear_model_predict_regression(IntPtr model, IntPtr inputs, int input_dim);
    
    [DllImport(DLL)]
    public static extern float linear_model_predict_classification(IntPtr model, IntPtr inputs, int input_dim);

    [DllImport(DLL)]
    public static extern float linear_model_train_classification(IntPtr model, IntPtr dataset_inputs, int dataset_inputs_size, int input_dim, IntPtr expected_outputs,  int iterations_count,  double alpha);

#else

    public static  int[] linear_model_create(int input_dim) { 
        UnityEngine.Debug.Log("linear_model_create");
        return new int[2] { 0, 1 };
    }


    public static  float linear_model_predict_regression(int[] model, int[] inputs) {
        UnityEngine.Debug.Log("linear_model_predict_regression");
        return 0f;
    }
    
    public static  float linear_model_predict_classification(int[] model, int[] inputs)
    {
        UnityEngine.Debug.Log("linear_model_predict_classification");
        return 0f;
    }


    public static  void  linear_model_train_classification(int[] model, int[] dataset_inputs, int[] dataset_expected_outputs, int iterations_count,  float alpha, bool should_plot_results, int plot_every_n_steps)
    {
        UnityEngine.Debug.Log("linear_model_train_classification");
    }

#endif
}