import requests, time, sys

# Upload resume PDF
with open(r'C:\Users\Bhavya Jain\Desktop\Bhavya Jain_RESUME.pdf', 'rb') as f:
    resp = requests.post('http://localhost:8000/upload', files={'file': ('resume.pdf', f, 'application/pdf')})

print('Upload status:', resp.status_code)
print('Response:', resp.json())

print('\nPolling for ready status...')
last_msg = ''
for i in range(180):
    time.sleep(2)
    docs = requests.get('http://localhost:8000/documents').json().get('documents', [])
    for doc in docs:
        status = doc.get('status', '')
        msg = doc.get('status_message', '')
        if msg != last_msg:
            print(f'  [{i*2}s] {status.upper()} | {msg[:100]}')
            last_msg = msg
        if status == 'ready':
            print('\nSUCCESS! Document indexed.')
            print('  doc_id   =', doc['doc_id'])
            print('  chunks   =', doc.get('num_chunks'))
            print('  links    =', doc.get('num_links'))
            print('  crawled  =', doc.get('num_crawled'))
            sys.exit(0)
        if status == 'error':
            print('\nERROR:', msg)
            sys.exit(1)

print('Timed out waiting for document to be ready')
