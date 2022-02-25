using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class CameraMove: MonoBehavior
{
    Transform playerTransform;
    Vector3 Offset;

    void Awake(){
        playerTransform=GameObject.FindGameObjectWithTag("Player").transform;
        Offset = transform.position = playerTransform.position;
    }

    void LateUpdate()
    {
        transform.position = Offset + playerTransform.position;
    }
}