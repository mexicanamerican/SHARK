import logging
import traceback
import os
import logging
import traceback
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

if __name__ == "__main__":
    if args.app == "txt2img":
        txt2img.main()
    elif args.app == "img2img":
        img2img.main()
    #   elif args.app == "inpaint":
    #       inpaint.main()
    #   elif args.app == "outpaint":
    #       outpaint.main()
    else:
        if os.path.exists('github_actions_logs.txt'):
            print('Error logs file exists. Please review the file for more information about the failure.')
        else:
            print('No error logs file found.')
