# Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
import sys
sys.path.append(os.getcwd)
# try:
# 	os.chdir(os.path.join(os.getcwd(), 'utils'))
# 	print(os.getcwd())
# except:
# 	pass

import requests
#from requests_toolbelt import MultipartEncoder
from bs4 import BeautifulSoup

LOGIN_URL = 'https://recsys.trivago.cloud/team/?login=1'
LOGOUT_URL = 'https://recsys.trivago.cloud/team/?logout=1'
SUB_URL = 'https://recsys.trivago.cloud/submission/'

HEADERS = {
  'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36',
  'Connection': 'keep-alive',
  'Host': 'recsys.trivago.cloud',
  'Accept-Language': 'it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7'
}

def __success(response):
  return response.status_code >= 200 and response.status_code < 300

def __login(username='policloud8@gmail.com', password='FerrariDC8'):
  # Return a session logged into the ACM competition website

  # begin a new session
  session = requests.session()
  # get the login tokens
  try:
    res = session.get(LOGIN_URL, headers=HEADERS)
  except:
    return None

  if not __success(res):
    return None

  html = BeautifulSoup(res.text, 'html.parser')
  form = html.find_all('form', id='LoginRegisterLoginForm')[0]
  hidden_inputs = form.find_all('input', type='hidden')
  
  # build the login POST request body
  post_args = [ ('login_name', username), ('login_pass', password), ('login_submit', 'Login') ]
  for t in hidden_inputs:
    post_args.append((t['name'],t['value']))
  
  # send the POST request
  try:
    r1 = session.post(LOGIN_URL, data=post_args, headers=HEADERS)
  except:
    return None

  if not __success(r1):
    return None
  
  if 'Team Description and Members' not in r1.text:
    return None
  
  return session

def __send_sub(file_path, session):
  filename = file_path.split('/')[-1]

  try:
    # m = MultipartEncoder(fields={
    #   'solution': (filename, open(file_path,'rb'), 'text/csv'),
    #   'submit': ''
    # })

    #print(m.to_string().decode('ascii'))
    payload = { 'solution': (filename, open(file_path, 'rb'), 'text/csv'), 'submit':'' }
    res = session.post(SUB_URL, files=payload, data={'submit':''}, headers=HEADERS)
    #res = session.post(SUB_URL, data=m, headers=HEADERS)
  except:
    return None

  if not __success(res):
    return None
  
  print(res.text)   #debug
  html = BeautifulSoup(res.text, 'html.parser')
  success_div = html.select('div.alert.alert--success')
  error_div = html.select('div.alert.alert--error')
  
  if len(success_div) == 0:  # something went wrong
    message = error_div[0].find('p', {'class':'alert__message'}).get_text()
    print()
    print(message)
    print()
    return None
  
  return session

def __logout(session):
  session.get(LOGOUT_URL, headers=HEADERS)
  return 


def send(file_path, username='policloud8@gmail.com', password='FerrariDC8'):
  session = __login(username, password)
  if session is None:
    print('Error while logging in!')
    return False

  session = __send_sub(file_path, session)
  if session is None:
    return False
  
  __logout(session)
  return True
  
    

#if __name__ == "__main__":
  #send('dataset/original/submission_popular.csv', username='keyblade95@live.it', password='p@ssword123')
  #send('/Users/federico/Desktop/sub.csv', username='keyblade95@live.it', password='p@ssword123')
  #print('NOTHING TO DO')
