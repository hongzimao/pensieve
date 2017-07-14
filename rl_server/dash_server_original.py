#!/usr/bin/env python
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import SocketServer
import base64
import urllib
import sys
import json

class Request_Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = json.loads(self.rfile.read(content_length))
        print post_data
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        # Hongzi: this is where the RL algorithm will be called (rather than always returning a quality of '1')
        data = '1'
        self.send_header('Content-Length', len(data) )
        self.send_header('Access-Control-Allow-Origin', "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        print >> sys.stderr, 'GOT REQ'
        self.send_response(200)
        #self.send_header('Cache-Control', 'Cache-Control: no-cache, no-store, must-revalidate max-age=0')
        self.send_header('Cache-Control', 'max-age=3000')
        self.send_header('Content-Length', 20)
        self.end_headers()
        self.wfile.write("console.log('here');")


    def log_message(self, format, *args):
        return

def run(server_class=HTTPServer, handler_class=Request_Handler, port=8333):
    server_address = ('localhost', port)
    httpd = server_class(server_address, handler_class)
    print 'Listening on port ' + str(port)
    httpd.serve_forever()

if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
