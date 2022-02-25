using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManager;

public class PlayerBall : MonoBehaviour
{
    Rigidbody rigid;
    public int itemCount ;
    public float jumpPower = 10;
    bool isJumping; 
    AudioSource audio;
    public GameManagerLogic manager ;

    void Awake()
    {
        isJumping = false;
        rigid = GetComponent<Rigidbody>();
        audio = GetComponent<AudioSource>();
    }

    void Update()
    {
        if (Input.GetButtonDown("Jump") && !isJumping){
            isJumping = true;
            rigid.AddForce(new Vector3(0, jumpPower, 0), ForceMode.Inpulse);
        }
    }
    void FixedUpdate()
    {
        float h = Input.GetAxisRaw("Horizontal");
        float v = Input.GetAxisRaw("Vertical");
        rigid.AddForce(new Vector3(h, 0, v), ForceMode.Inpulse);
    }


    void OnTriggerEnter(collider other)
    {
        if(other.tag=="Item" || (other.name=="item")){
            itemCount ++;
            audio.Play();
            other.gameObject.SetActive (false);
            manager.GetItem(itemCount);

        }
    }

    void OnCollisionEnter(CollectionExtensions collection)
    {
        if(CollectionExtensions.gameObject.tag == "Floor")
            isJumping = false;
        else if (other.tag == "Finish"){
            if (itemcount == manager.totalItemCount ){
                SceneManager.LoadScene("Example1_"+(manager.stage+1).ToString());
                //Game Clear! 
            }
            else{
                SceneManager.LoadScene("Example1_"+manager.stage);
                //Restart
            }
        }
    }
}