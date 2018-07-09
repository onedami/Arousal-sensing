from __future__ import division, print_function
import pandas as pd
import math
import seaborn as sns
import scipy.stats
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.integrate import simps
from numpy import trapz
from collections import Counter
pd.options.mode.chained_assignment = None  # default='warn'
class Analysis:
    pupil = pd.DataFrame()
    reported_feedback = pd.DataFrame()
    all_tasks = pd.DataFrame()
    comparison_result = pd.DataFrame()
    collation_result = pd.DataFrame()
    normalized_result = pd.DataFrame()


# read pupil data from file
    def readPupilData(self,filename):
        pupil_data = pd.read_table(filename, sep='\t', encoding="utf-8-sig")
        # pupil_data =  pd.read_csv(filename)
        pupil_data = pupil_data.drop_duplicates()
        pupil_data_AOI = pupil_data.filter(regex=("AOI.*"))
        pupil_data = pupil_data.loc[:,
                     ['ParticipantName', 'MediaName', 'RecordingTimestamp', 'MouseEventIndex', 'MouseEvent',
                      'KeyPressEventIndex', 'KeyPressEvent', 'GazeEventType', 'PupilLeft', 'PupilRight', 'ValidityLeft',
                      'ValidityRight']]
        Analysis.pupil = pd.concat([pupil_data, pupil_data_AOI], axis=1)


# helper method to convert the data into desired format
    def fill_aoi(self):
        pupil_data = self.pupil
        all_aois = pupil_data.filter(regex=("AOI.*")).columns
        pupil_data['AOI'] = 0
        start = 'AOI['
        end = ']'
        prev_aoi = 0
        aoi = 0
        for aoi_column in all_aois:
            aoi = aoi_column[aoi_column.find(start) + len(start):aoi_column.rfind(end)]
            pupil_data.loc[pupil_data[aoi_column] == 1, 'AOI'] = aoi
        # delete all other AOI columns because they are now redundant
        pupil_data.drop(all_aois, axis=1, inplace=True)
        # remove extra spaces and numbers after AOI name.
        # fill all Nan to 0
        pupil_data['AOI'].fillna(value=0, inplace=True)
        # specific to picture study
        # convert media names to picture names
        pupil_data['MediaName'] = pupil_data['MediaName'].map(
            {'7010.jpg': 'Basket', '7175.jpg': 'Lamp', '2440.jpg': 'Boy',
             '2513.jpg': 'Woman', '2312.jpg': 'Woman_baby', '2359.jpg': 'Mother_baby',
             '8231.jpg': 'Boxer', '9031.jpg': 'Dirt', '1302.jpg': 'Dog',
             '4597.jpg': 'Couple', '1321.jpg': 'Bear', '8492.jpg': 'Roller_coaster',
             })

        Analysis.pupil = pupil_data


    def interpolate(self):
        # convert to series
        pupil_data = Analysis.pupil
        sleft = pd.Series(pupil_data.PupilLeft)
        sright = pd.Series(pupil_data.PupilRight)
        # linear interpolation
        sleft = sleft.interpolate()
        sright = sright.interpolate()
        # convert back to dataframe
        dleft = pd.DataFrame(sleft)
        dright = pd.DataFrame(sright)
        # drop previous values of pupil
        pupil_data = pupil_data.drop('PupilLeft', 1)
        pupil_data = pupil_data.drop('PupilRight', 1)
        # append with new values after interpolation
        pupil_data = pd.concat([pupil_data, dleft, dright], axis=1)
        pupil_data['PupilLeft'].fillna(pupil_data['PupilLeft'].mean(), inplace=True)
        pupil_data['PupilRight'].fillna(pupil_data['PupilRight'].mean(), inplace=True)
        Analysis.pupil = pupil_data


# participant is a string argument of the participant ID to be extracted.
    def get_participant(self, participant):
        pupil_data = Analysis.pupil
        participant = pupil_data.loc[pupil_data.ParticipantName == participant]
        participant = participant.reset_index()
        return participant


# Extract the data of a single media (stimulus) contained in the pupil_data
    def extractMedia(self, pupil_data, media):
        participant_x_medium = pupil_data[pupil_data['MediaName'] == media]
        return participant_x_medium


    def create_aggregated_average(self, pupil_data, window_size):
        aoi_selected = 0
        # pupil_data['left_window'] = 0
        # pupil_data['right_window'] = 0
        pupil_data.loc[:, 'left_window'] = 0
        pupil_data.loc[:, 'right_window'] = 0
        aggregated_pupil_data = pd.DataFrame(columns=pupil_data.columns)
        for current_index in range(1, len(pupil_data) - 1, window_size):
            temp_pupil_data = pupil_data.iloc[current_index:current_index + window_size - 1]
            aggregated_pupil_data = aggregated_pupil_data.append(temp_pupil_data.iloc[0])
            data = Counter(temp_pupil_data['AOI'])

            # data.most_common()   # Returns all unique items and their counts
            aoi = data.most_common(2)  # Returns the two most frequently occurring item
            aoi_selected = aoi[0][0]
            if (aoi_selected == 0 or aoi_selected == '0') and len(aoi) > 1:
                aoi_selected = aoi[1][0]  # if the topmost is 0, get second most common AOI if it exists
            if (aoi_selected == 0 or aoi_selected == '0') and len(aggregated_pupil_data) > 2:
                aoi_selected = aggregated_pupil_data['AOI'].iloc[len(aggregated_pupil_data) - 2]
            aggregated_pupil_data['left_window'].iloc[len(aggregated_pupil_data) - 1] = temp_pupil_data[
                'PupilLeft'].mean()
            aggregated_pupil_data['right_window'].iloc[len(aggregated_pupil_data) - 1] = temp_pupil_data[
                'PupilRight'].mean()
            aggregated_pupil_data['AOI'].iloc[len(aggregated_pupil_data) - 1] = aoi_selected
            aggregated_pupil_data = aggregated_pupil_data.reset_index(drop=True)
        return aggregated_pupil_data

# convert pupil data into desired number of levels
    def convert_to_scale(self, pupil_data, levels):
        levels = levels - 1
        left_mean = pupil_data['PupilLeft'].mean()
        right_mean = pupil_data['PupilRight'].mean()
        left_span = pupil_data['left_window'].max() - pupil_data['left_window'].min()
        right_span = pupil_data['right_window'].max() - pupil_data['right_window'].min()
        left_unit = left_span / levels
        right_unit = right_span / levels
        pupil_data.loc[:, 'left_level'] = 0
        pupil_data.loc[:, 'right_level'] = 0
        pupil_data.loc[:, 'pupil_level'] = 0
        for current_index in range(0, len(pupil_data.index)):
            left_level = round((pupil_data['left_window'].iloc[current_index] - left_mean) / left_unit)
            right_level = round((pupil_data['right_window'].iloc[current_index] - right_mean) / right_unit)
            pupil_data['left_level'].iloc[current_index] = left_level
            pupil_data['right_level'].iloc[current_index] = right_level
            pupil_data['pupil_level'].iloc[current_index] = (left_level + right_level) / 2
        # uncoment 2 lines below to convert to absolute levels (non-negative)
        # minimum = pupil_data['left_level'].min() * -1
        # pupil_data['left_level'] = pupil_data['left_level'] + minimum
        return pupil_data

# takes pupil measures and return indices of all peaks
    def detect_peaks(self, pupils):
        pupils = pupils.values
        peak_indices = pd.Series()
        # start from the 2nd element (because 1st element cannot be a peak) to the second to the last element(because last element cannot be a peak)
        for index in range(1, len(pupils) - 1):
            # it's a peak if the current element is greater than previous and greater or equal to next
            if ((pupils[index] > pupils[index - 1]) & (pupils[index] >= pupils[index + 1])):
                peak_indices = peak_indices.append(pd.Series(index))
        peak_indices = peak_indices.reset_index(drop=True)
        return peak_indices

# takes pupil measures and return indices of all peaks
    def detect_peaks_(self, pupils):
        pupils = pupils.values
        peak_indices = pd.Series()
        # start from the 2nd element (because 1st element cannot be a peak) to the second to the last element(because last element cannot be a peak)
        for index in range(1, len(pupils) - 1):
            # it's a peak if the current element is greater than previous and greater or equal to next
            if ((pupils[index] > pupils[index - 1]) & (pupils[index] >= pupils[index + 1])):
                if (len(peak_indices) > 0 and pupils[peak_indices.iloc[-1]:index+1].min() >= pupils[peak_indices.iloc[-1]]):
                    peak_indices = peak_indices.drop(index=len(peak_indices) - 1)
                peak_indices = peak_indices.append(pd.Series(index))

        peak_indices = peak_indices.reset_index(drop=True)
        return peak_indices


    def get_peaks(self,participant_data,pupil):
        if pupil == 'r':
            column = 'right_level'
        elif pupil == 'l':
            column = 'left_level'
        else:
            column = 'pupil_level'
        return self.detect_peaks(participant_data[column])


    def get_change_time(self, events, likelihood_of_change, cut_off):
        change_timestamps = np.array([])
        previous = 0
        for i in range(len(likelihood_of_change) - 1):
            if likelihood_of_change[i] > cut_off:
                if (events.iloc[i].RecordingTimestamp - events.iloc[previous].RecordingTimestamp > 2000):
                    change_timestamps = np.append(change_timestamps, events.iloc[i].RecordingTimestamp)
                    previous = i
        return change_timestamps


    def extract_nth_task(self, change_timestamps, n):
        pupil_data = self.pupil
        if (n == -1):  # extract data from first index (0) to the index where RecordingTimestamp = n
            start_index = 0
            end_index = pupil_data[pupil_data['RecordingTimestamp'] == change_timestamps[0]].index[0]
            temp = pupil_data.iloc[start_index:end_index]
        elif n == len(change_timestamps) - 1:
            start_index = pupil_data[pupil_data['RecordingTimestamp'] == change_timestamps[n]].index[0]
            end_index = pupil_data['RecordingTimestamp'].idxmax()
            temp = pupil_data.iloc[start_index:end_index]
        elif n > len(change_timestamps):
            return "End_of_change"
        else:
            start_index = pupil_data[pupil_data['RecordingTimestamp'] == change_timestamps[n]].index[0]
            end_index = pupil_data[pupil_data['RecordingTimestamp'] == change_timestamps[n + 1]].index[0]
            temp = pupil_data.iloc[start_index:end_index]
        return temp


    def scale_tasks(self, change_timestamps):
        pupil_data = self.pupil
        pupil_data['task_name'] = ''
        all_tasks = pd.DataFrame(columns=pupil_data.columns)
        for n in range(-1, (len(change_timestamps))):
            task_number = n + 1
            task_n = extract_nth_task(pupil_data, change_timestamps, n)
            task_aggregate = create_aggregated_average(task_n, 50)
            task_scale = convert_to_scale(task_aggregate, 7)
            task_scale['task_name'] = 'Task_' + str(task_number)
            all_tasks = all_tasks.append(task_scale)
        all_tasks = all_tasks.reset_index(drop=True)
        return all_tasks


# loop through all peaks and get the cause
    def get_peak_cause(self, peak_indices, pupil_data, pupil):
        if pupil == 'r':
            column = 'right_level'
        elif pupil == 'l':
            column = 'left_level'
        else:
            column = 'pupil_level'
        participant_x = pupil_data
        participant_x = participant_x.reset_index(drop=True)
        peak_cause = pd.DataFrame(
            columns=['peak_index', 'foot', 'footAOI', 'footDelta', 'valley', 'valleyAOI', 'valleyDelta',
                     'peakDuration'])
        for current_peak in peak_indices:
            # valley = 0
            foot = current_peak - 1
            valley = foot
            for point in range((current_peak - 1), -1, -1):
                if (participant_x[column].ix[point] < participant_x[column].ix[current_peak]
                        and participant_x[column].ix[point] < participant_x[column].ix[valley]):
                    valley = point
                    continue
                elif (participant_x[column].ix[point] > participant_x[column].ix[valley]):
                    break
            footAOI = str(participant_x.AOI.ix[foot])
            footDelta = abs(participant_x[column].ix[current_peak] - participant_x[column].ix[foot])
            valleyAOI = str(participant_x.AOI.ix[valley])
            valleyDelta = abs(participant_x[column].ix[current_peak] - participant_x[column].ix[valley])
            peakDuration = current_peak - valley
            peak_cause.loc[len(peak_cause)] = [current_peak, foot, footAOI, footDelta, valley, valleyAOI, valleyDelta,
                                               peakDuration]
        return peak_cause

    def collate_peaks(self, all_tasks):
        peaks = pd.DataFrame()
        for task in all_tasks.task_name.unique():
            # print(task)
            this_task = ''
            duration = 0
            this_task = all_tasks[all_tasks['task_name'] == task]
            participant = this_task.ParticipantName.iloc[0]
            task_name = task
            duration = this_task.RecordingTimestamp.max() - this_task.RecordingTimestamp.min()
            if duration == 0:  # duration is 0 if there is only one record, max will be equal to min
                print(task_name)
                continue
            if duration < 3000:
                continue
            peak_indices = self.get_peaks(this_task, 'l')
            peak_causes = self.get_peak_cause(peak_indices, this_task, 'l')
            peak_causes['ParticipantName'] = participant
            peak_causes['task'] = task_name
            peaks = peaks.append(peak_causes)
        return peaks

    def compare_tasks(self, all_tasks):
        comparison = pd.DataFrame(
            columns=['ParticipantName', 'task_name', 'AOI', 'min', 'max', 'mean', 'std', 'duration', 'number_of_peaks',
                     'peak_density', 'peak_indices', 'peak_feet', 'peak_valley', 'footDelta', 'valleyDelta',
                     'peakDuration'])
        for task in all_tasks.task_name.unique():
            # print(task)
            this_task = ''
            AOI = 0
            min, max, mean, std, duration, number_of_peaks, peak_density = 0, 0, 0, 0, 0, 0, 0
            this_task = all_tasks[all_tasks['task_name'] == task]
            participant = this_task.ParticipantName.iloc[0]
            task_name = task
            min = this_task.left_level.min()
            max = this_task.left_level.max()
            mean = this_task.left_level.mean()
            std = this_task.left_level.std()
            duration = this_task.RecordingTimestamp.max() - this_task.RecordingTimestamp.min()
            if duration == 0:  # duration is 0 if there is only one record, max will be equal to min
                print(task_name)
                continue
            if duration < 3000:
                continue
            peak_indices = self.get_peaks(this_task, 'l')
            peak_causes = self.get_peak_cause(peak_indices, this_task, 'l')
            peak_feet = ','.join(peak_causes.footAOI)
            peak_valley = ','.join(peak_causes.valleyAOI)
            footDelta = peak_causes.footDelta.sum()
            valleyDelta = peak_causes.valleyDelta.sum()
            number_of_peaks = len(peak_indices)
            peakDuration = peak_causes.peakDuration.sum()
            peak_density = (number_of_peaks / duration)
            AOI = ','.join(this_task.AOI.astype('str'))
            comparison.loc[len(comparison)] = [participant, task_name, AOI, min, max, mean, std, duration,
                                               number_of_peaks, peak_density, peak_indices, peak_feet, peak_valley,
                                               footDelta, valleyDelta, peakDuration]
        return comparison


# grouping into tasks
    def groupTasks(self):
        pupil_data = Analysis.pupil
        all_tasks = pd.DataFrame(columns=pupil_data.columns)
        all_tasks['task_name']=''
        for participant in pupil_data.ParticipantName.unique():
            participant_x = self.get_participant(participant)
            participant_x = participant_x.sort_values(by='RecordingTimestamp')
            #count = 0
            #if (self.reported_feedback[self.reported_feedback['Participant']==participant].Eye_tracker_accuracy.iloc[0] <= 70):
                #continue
            uniqueTasks = participant_x.MediaName.unique()
            for task in uniqueTasks:
                #count = count + 1
                #if count > 5:
                #if (count < 5) | (count > 8):
                #if count < 9:
                    #continue
                if(task == task):
                    participant_x_task = self.extractMedia(participant_x,task)
                    #We can only use observations greater than 3seconds
                    if((participant_x_task.RecordingTimestamp.max() - participant_x_task.RecordingTimestamp.min()) < 3000):
                        continue
                    participant_x_aggregate = self.create_aggregated_average(participant_x_task,50)
                    #curr_task = participant_x_aggregate
                    participant_x_scale = self.convert_to_scale(participant_x_aggregate,7)
                    participant_x_scale['task_name'] = task
                    all_tasks = all_tasks.append(participant_x_scale)
        Analysis.all_tasks = all_tasks


# loop through all participnts and extract features
    def compare_results(self):
        all_tasks = Analysis.all_tasks
        reported_feedback = self.reported_feedback
        comparison_result = pd.DataFrame(columns=['ParticipantName','task_name','AOI','min','max',
                                              'mean','std','start_time','end_time','duration','number_of_peaks','peak_density',
                                              'peak_indices','peak_feet','peak_valley','footDelta','valleyDelta'])
        for participant in all_tasks.ParticipantName.unique():
            #print(participant)
            participant_result = self.compare_tasks(all_tasks[all_tasks['ParticipantName']==participant])
            comparison_result = comparison_result.append(participant_result)
        Analysis.comparison_result = comparison_result

# loop through all participnts and extract features
    def collate_results(self):
        all_tasks = Analysis.all_tasks
        collation_result = pd.DataFrame()
        for participant in all_tasks.ParticipantName.unique():
            # print(participant)
            participant_result = self.collate_peaks(all_tasks[all_tasks['ParticipantName'] == participant])
            collation_result = collation_result.append(participant_result)
        Analysis.collation_result = collation_result


# Normalize variables for each participant
    def normalize_result(self):
        comparison_result = Analysis.comparison_result
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(1,3))
        normalized_result = pd.DataFrame(columns=comparison_result.columns)
        normalized_result['peakDuration_normalized'] = 0
        normalized_result['footDelta_normalized'] = 0
        normalized_result['valleyDelta_normalized'] = 0
        normalized_result['duration_normalized'] = 0
        for participant in comparison_result.ParticipantName.unique():
            participant_record = comparison_result.loc[comparison_result['ParticipantName']==participant]
            participant_record['peakDuration_normalized'] = \
                min_max_scaler.fit_transform(participant_record[['peakDuration']]/1000)
            participant_record['footDelta_normalized'] = min_max_scaler.fit_transform(participant_record[['footDelta']])
            participant_record['valleyDelta_normalized'] = \
                min_max_scaler.fit_transform(participant_record[['valleyDelta']])
            participant_record['duration_normalized'] = \
                min_max_scaler.fit_transform(participant_record[['duration']])
            normalized_result = normalized_result.append(participant_record)
        Analysis.normalized_result = normalized_result


    def generate_inference(self):
        normalized_result = Analysis.normalized_result
        normalized_result['cumulative'] = 0
        for participant in normalized_result.ParticipantName.unique():
            for task in normalized_result.task_name.unique():
                '''peakDuration = normalized_result.loc[
                    (normalized_result['ParticipantName'] == participant) & (normalized_result['task_name'] == task)][
                    'peakDuration_normalized']
                foot = normalized_result.loc[
                    (normalized_result['ParticipantName'] == participant) & (normalized_result['task_name'] == task)][
                    'footDelta_normalized']'''
                duration = normalized_result.loc[
                    (normalized_result['ParticipantName'] == participant) & (normalized_result['task_name'] == task)][
                    'duration_normalized']
                valley = normalized_result.loc[
                    (normalized_result['ParticipantName'] == participant) & (normalized_result['task_name'] == task)][
                    'valleyDelta_normalized']
                #cumulative = ((foot) + (valley)) * (duration / 1000)
                cumulative = valley * duration
                normalized_result.loc[(normalized_result['ParticipantName'] == participant) & (
                            normalized_result['task_name'] == task), 'cumulative'] = cumulative
        Analysis.normalized_result = normalized_result


pictureStudy = Analysis()
pictureStudy.readPupilData('C:\\Users\\Chapy\\Desktop\\Olu desktop\\My Studies\\Picture study\\extracted_data.tsv')
pictureStudy.reported_feedback = pd.read_csv\
    ('C:\\Users\\Chapy\\Desktop\\Olu desktop\\My Studies\\Picture study\\just_reported_feedback.csv')
pictureStudy.fill_aoi()
pictureStudy.interpolate()
pictureStudy.groupTasks()
pictureStudy.compare_results()
pictureStudy.collate_results()
pictureStudy.normalize_result()
pictureStudy.generate_inference()
normalized_result = pictureStudy.normalized_result
collation_result = pictureStudy.collation_result

# Perform correlation test - participants reported arousal against the cumulative result of the algorithm
print("n = ",len(normalized_result))
print("participants = ", len(normalized_result.ParticipantName.unique()))
for participant in normalized_result.ParticipantName.unique():
    print(participant, len(normalized_result[normalized_result['ParticipantName']==participant]), normalized_result[normalized_result['ParticipantName']==participant].cumulative.mean())
