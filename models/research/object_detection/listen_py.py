#! /usr/bin/env python
# -*- coding: utf-8 -*-
import asyncio
import base64
import logging
import os
import shutil
import sys
from datetime import datetime
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
WATCH_PATH = '/home/zyan11/Downloads/models/research/object_detection/test_images/'
class FileMonitorHandler(FileSystemEventHandler):
 def __init__(self, **kwargs):
  super(FileMonitorHandler, self).__init__(**kwargs)
  
  self._watch_path = WATCH_PATH
 
 def on_modified(self, event):
  os.system("python3 webPro.py")
if __name__ == "__main__":
 event_handler = FileMonitorHandler()
 observer = Observer()
 observer.schedule(event_handler, path=WATCH_PATH, recursive=True) 
 observer.start()
 observer.join()