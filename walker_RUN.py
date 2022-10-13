from os import system 

s1 = "python main.py --env_name Walker2d-v3 --trial 0"
s2 = "python main.py --env_name Walker2d-v3 --trial 1"
s3 = "python main.py --env_name Walker2d-v3 --trial 2"

system(s1)
system(s2)
system(s3)