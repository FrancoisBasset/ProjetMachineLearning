using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class UpTest : MonoBehaviour
{
    // Start is called before the first frame update
    public Transform[] trainSpheresTransforms;

    public Transform[] testSpheresTransforms;

    public void TrainAndTest()
    {
        Debug.Log("Training and Testing");

        // Créer dataset_inputs
        // Créer dataest_expected_outputs

        // Create Model

        // Train Model

        // For each testSphere : Predict 
        foreach (var testSpheresTransform in testSpheresTransforms)
        {
            bool isUped = IsUped();
            Debug.Log($" --> {isUped}");
            if (isUped)
            {
                testSpheresTransform.position += Vector3.up * 2;
            }

        }
    }

    private bool IsUped()
    {
        var random = new System.Random();

        return (random.Next() > 0) ;
    }
}