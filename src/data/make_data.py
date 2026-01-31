import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../../')))

from src.utils.config import RAW_DATA_PATH,PROCESSED_DATA_PATH,DATETIME_COL,TARGET_COL
from src.utils.common import setup_logger

logger=setup_logger()

def main():
    logger.info('Loading raw data...')
    if not os.path.exists(RAW_DATA_PATH):
        logger.error(f'File Not found:{RAW_DATA_PATH}')
        return
    df=pd.read_csv(RAW_DATA_PATH)
    logger.info(f'Original Shape {df.shape}')

    df=df[df['Booking Status']=='Completed']

    logger.info(f'Shape after CCA{df.shape}')

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH),exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH,index=False)
    logger.info(f'Proessed data saved to {PROCESSED_DATA_PATH}')
if __name__=='__main__':
    main()
