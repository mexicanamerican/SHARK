import logging
import traceback
import os
import logging
import traceback
import sys
from apps.stable_diffusion.src import args
from apps.stable_diffusion.scripts import (
    img2img,
    txt2img,
    #    inpaint,
    #    outpaint,
)

try:
    if args.app == "txt2img":
        txt2img.main()
    elif args.app == "img2img":
        img2img.main()
    else:
        raise ValueError("Invalid app name")
except Exception as e:
    with open('github_actions_logs.txt', 'a') as log_file:
        log_file.write(f'Error: {str(e)}\n')
        log_file.write(f'Traceback: {traceback.format_exc()}\n')
    print('An error occurred. Please review the error logs file for more information.')
