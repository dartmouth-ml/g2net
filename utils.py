import datetime

def get_datetime_version():
    now = datetime.datetime.now()
    res = f"{now.month}{now.day}{now.year}_{now.hour}{now.minute}{now.second}"
    return res