def get_prediction(file):
    if str(file.filename).count('me') > 0:
        return 'meningioma'
    elif str(file.filename).count('gl') > 0:
        return 'glioma'
    elif str(file.filename).count('pi') > 0:
        return 'pituitary'
    else:
        return 'no tumor'
    
