# KumbayUni
WORK IN PROGRESS

Anonmyization feature is complete: to check it out, do the following
1. Add the file to assets/test_data/vid
2. Edit the main function in src/anon.py to open assets/test_data/vid/[YOUR_FILE]
3. Run with python3 src/anon.py

Website is not hosted yet, but you can run locally, using the command "flask run"

Important notes about flask and stuff.

I've made it so that you can run it using "flask run" from the home directory. The way
you toggle this, is by setting the environment variable FLASK_APP equal to the location of
app.py relative to where you want to run the program from; eg: export FLASK_APP=src/app
