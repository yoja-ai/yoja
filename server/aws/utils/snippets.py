import cryptography

# get the key from ServiceConf
fky = cryptography.fernet.Fernet(key_from_service_conf)
# encrypt the email
encrypted_email:bytes = fky.encrypt("email_address".encode())    
cookie = f'yoja-user={encrypted_email.decode()};another_cookie_name=another_cookie_val'


