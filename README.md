# fictional-tag
OpenAI gymnasium custom env for mujoco simulation with ball targeting a box, the target.

To Recreate, make sure to structures the files on the github repo in the following order in your project folder:


''''
'''
v Project Folder

  main.py
  v bexa
      setup.py
      readme.py
      v bexb
         __init__.py
         v envs
            __init__.py
            MyBallEnv.py
            v modelos
               helloworld.xml
'''               
''''
      
         


Notes:
1. Make sure to change line 30 of MyBallEnv with the absolute full path to your XML file
2. To import file, navigate terminal to project folder bexb and setup.py are in and run "pip install -e ."
3. Side Note: main.py does not have to be in the same folder as bexb or setup.py
