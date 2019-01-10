#!/usr/bin/env python

import sys
import os
import time
import atexit
from signal import SIGTERM

class Daemon(object):
    def __init__(self, pidfile, startfunc , stdin='/dev/null', stdout='/dev/null', stderr='/dev/null'):
        self.stdout = stdout
        self.stdin = stdin
        self.stderr = stderr
        self.pidfile = pidfile
        self.startfunc = startfunc

    def daemonize(self):
        """
         do the UNIX double-fork magic, see Stevens' "Advanced
                Programming in the UNIX Environment" for details (ISBN 0201563177)
                http://www.erlenstar.demon.co.uk/unix/faq_2.html#SEC16
        """
        try:
            pid = os.fork()
            if pid > 0:
                #Exit With non error
                sys.exit(0)
        except OSError as e:
            sys.stderr.write("fork #1 failed: %d (%s)\n" %
                             (e.errno, e.strerror))
            sys.exit(1)

        #decouple from second parrent
        os.chdir('/')
        os.setsid()
        os.umask(0)

        #Try do second fork
        try:
            pid = os.fork()
            if pid > 0:
                #exit from second parrent
                sys.exit(0)
        except OSError as e:
            sys.stderr.write('fork #2 failed: %d (%s) \n' %
                             (e.errno, e.strerror))
            sys.exit(1)
        # redirect from file descriptors.
        sys.stdout.flush()
        sys.stderr.flush()
        si = open(self.stdin, 'r')
        so = open(self.stdout, 'a+')
        se = open(self.stderr, 'a+', 0)

        os.dup2(si.fileno(), sys.stdin.fileno())
        os.dup2(so.fileno(), sys.stdout.fileno())
        os.dup2(se.fileno(), sys.stderr.fileno())

        # Write pidfile
        pid = str(os.getpid())
        with open(self.pidfile, 'w+') as f:
            f.write(pid + '\n')

    def delpid(self):
        os.remove(self.pidfile)


    def start(self):
        """ 
            Start the damon
        """
        #check if we got a valid pidfile
        try:
            with open(self.pidfile, 'r') as pf:
                pid = int(pf.read().strip())

        except IOError as e:
            pid = None
            print('Error in opening file\n Message: %s',e.strerror )

        if pid:
            message = "pidfile {0} already exist. " + \
                      "Daemon already running?\n"
            sys.stderr.write(message.format(self.pidfile))
            sys.exit(1)

        self.daemonize()
        self.run(self.startfunc)


    def stop(self):
		# Get the pid from the pidfile
        try:
            with open(self.pidfile, 'r') as pf:
                pid = int(pf.read().strip())
        except IOError:
            pid = None
            if not pid:
                message = "pidfile {0} does not exist. " + "Daemon not running?\n"
                sys.stderr.write(message.format(self.pidfile))
                return  # not an error in a restart

		# Try killing the daemon process
        try:
            while 1:
                os.kill(pid, SIGTERM)
                time.sleep(0.1)    
        except OSError as err:
            e = str(err.args)
            if e.find("No such process") > 0:
                if os.path.exists(self.pidfile):
                    os.remove(self.pidfile)
                else:
                    print(str(err.args))
                    sys.exit(1)


    def restart(self):
        """Restart the daemon."""
        self.stop()
        self.start()
        
    def run(self, callback):
        """You should override this method when you subclass Daemon.
		
		    It will be called after the process has been daemonized by 
		    start() or restart().
            """
        callback()