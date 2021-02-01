import numpy
import json

class Autograder:
    def __init__(self, name):
        self.results = {'output': name, 'tests': []}
        self.thresh = []

    def create_test(self, test_name, max_score, full_thresh=0, sixty_thresh=0):
        test_dict = {'max_score': max_score, 'name': test_name}
        self.results['tests'].append(test_dict)
        self.thresh.append([full_thresh, sixty_thresh])
        return len(self.results['tests']) - 1

    def score_test(self, test_id, val, feedback=''):
        test = self.results['tests'][test_id]
        if self.thresh[test_id][0] == 0 or type(val) == bool:
            #binary test
            if val:
                frac = 1
            else:
                frac = 0
            test['output'] = feedback
        else:
            #test on linear scale
            if self.thresh[test_id][0] < self.thresh[test_id][1]:
                #lower better
                frac = 1 - (val - self.thresh[test_id][0])/(self.thresh[test_id][1] - self.thresh[test_id][0])
            else:
                #higher better
                frac = (val - self.thresh[test_id][1])/(self.thresh[test_id][0] - self.thresh[test_id][1])
            test['output'] = 'Score: ' + str(val) + '\n' + '100%: ' + str(self.thresh[test_id][0]) + ', 60%: ' + str(self.thresh[test_id][1])
            if feedback != '':
                test['output'] += '\n' + feedback

            frac = max(0, min(1, frac)) #threshhold
            if frac > 0:
                frac = 0.6 + frac*0.4

        test['score'] = test['max_score'] * frac

    def write(self, path):
        with open(path, 'w') as results_file:
            json.dump(self.results, results_file)
