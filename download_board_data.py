from boardlaw import backup
import sys
import time
import b2sdk.v1 as b2

api    = backup.api('boardlaw')                     
bucket = api.get_bucket_by_name('boardlaw')         

# List only the top‐level entries under 'output/'
for file_info, name in bucket.ls(folder_to_list='output/pavlov/', recursive=False):
    run = name.strip('/').split('/')[-1]

    dest = 'local_storage'

    bucket = 'boardlaw'
    api = backup.api(bucket)

    syncer = b2.Synchronizer(4)
    with b2.SyncReport(sys.stdout, False) as reporter:
        syncer.sync_folders(
            source_folder=b2.parse_sync_folder(f'b2://boardlaw/output/pavlov/{run}', api),
            dest_folder=b2.parse_sync_folder(f'boardlaw-paper-v2/output/pavlov/{run}', api),
            now_millis=int(round(time.time() * 1000)),
            reporter=reporter)
    









# from pavlov import stats, storage, runs, files

# run = '2021-03-26 15-30-17 harsh-wait'

# # To list the runs you've downloaded
# runs.pandas()

# # To list the files downloaded for a specific run
# files.pandas(run)

# # To view the residual variance from the run
# stats.pandas(run, 'corr.resid-var')