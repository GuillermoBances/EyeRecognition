import os

# CLASS FOR SORTING FINAL RESULTS
class sorted_list():
    
    def __init__(self, max_len, order_position):
        self.max_len = max_len
        self.order_position = order_position
        self.current_len = 0
        self.check_wait = 0
        self.alert = False
        self.max_check_wait = 5
        self.iter = 0
        self.list = []

    def start_alert(self, n, dt):
        print('DROWSINESS DETECTION: ASLEEP FOR 3 SECONDS, STARTING AT ITERATION ' + str(n) + ', AND DATETIME ' + dt)
    
    # FUNCTION FOR STOP ALERT
    def end_alert(self, n, dt):
        print('DROWSINESS DETECTION: WAKEN UP AT ITERATION ' + str(n) + ', AND DATETIME ' + dt)

    # APPEND NEW RESULTS AND SORT THEM TO CHECK IN ORDER IF ANY IMAGE CONTAINE OPENED EYES  
    def append(self, value):
        self.iter += 1
        if self.current_len < self.max_len:
            self.current_len += 1
        else:
            self.list.pop(0)
            
        self.list.append(value)
        sorted(self.list, key=lambda x:x.split(";")[1])

        self.check_wait += 1
        if self.current_len >= self.max_len and self.check_wait >= self.max_check_wait:
            self.check_wait = 0
            asleep, dt = self.check_drowsy()
            if self.alert and not asleep:
                self.alert = False
                self.end_alert(self.iter, dt)
            else:
                if not self.alert and asleep:
                    self.alert = True
                    dt = self.list[0].split(";")[1]
                    self.start_alert(self.iter, dt)

    # CHECK IF ANY OF 90 RESULTS IS BOTH EYES ARE OPENED
    def check_drowsy(self):
        flag = True
        dt = ''
        for x in self.list:
            token = x.split(";")
            ocurr = token[2].count('OPENED')
            if ocurr == 2:
                dt = token[1]
                flag = False
                break
        return flag, dt
    
    def get_len(self):
        return len(self.list)    