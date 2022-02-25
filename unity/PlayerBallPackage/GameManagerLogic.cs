using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManager;
using UnityEngine.UI;

public class GameManagerLogic : MonoBehaviour
{
    public int totalItemCount;
    public int stage;
    public Text stageCountText;
    public Text playerCountText;
    
    void Awake()
    {
        stageCountText.text = "/ " + totalItemCount;
    }

    public void GetItem(int count)
    {
        playerCountText.text = count.ToString();
    }

    private void OnTriggerEnter(Collider other){
        if(other.gameObject.tag == "Player"){
            SceneManager.LoadScene(GameManagerLogic.stage);
        }

    }
}