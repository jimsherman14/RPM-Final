# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
from cProfile import label
from PIL import Image
import numpy as np
import cv2
import timeit


class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    def __init__(self):
        self.allg = {}
        self.xrange = {}
        self.yrange = {}
        self.i = 0
        self.isEmptySet = False

    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.
    def Solve(self, problem):
        # print(problem.name)
        start = timeit.default_timer()
        if problem.name != "Basic Problem E-09":
            return -1
        # {'A" : {0: [...], 1: [...], ...}} This holds the coordinates of the corners of each shape in each figure
        labeled_figures = self.find_shapes(problem.figures)
        # print("labeld")
        # print(labeled_figures)
        # holds semantic relationships between shapes within a figure. numcorners, corners, x-yrange, inside, rightof/leftof, above/below
        semantic_figures = self.generate_networks(labeled_figures)
        self.gather_pixel_ranges(semantic_figures)
        # print(self.xrange)
        # print(self.yrange)

        # print("sem")
        # print(semantic_figures)
        # print(labeled_figures['A'][0][2])
        scores = {}
        if problem.problemType == "2x2":
            a_to_b_transition = self.determineTranslation(
                semantic_figures['A'], semantic_figures['B'], self.allg['A'], self.allg['B'])  # last two parameters are converted grayscale matrices
            a_to_c_transition = self.determineTranslation(
                semantic_figures['A'], semantic_figures['C'], self.allg['A'], self.allg['C'])
            # print("trans")
            # print(a_to_b_transition)
            #print("ab above, ac below")
            # print(a_to_c_transition)
            for figure_id in range(1, 7):
                curr = semantic_figures[str(figure_id)]
                curr_gray = self.allg[str(figure_id)]
                c_to_d_transition = self.determineTranslation(
                    semantic_figures['C'], curr, self.allg['C'], curr_gray)
                b_to_d_transition = self.determineTranslation(
                    semantic_figures['B'], curr, self.allg['B'], curr_gray)
                score = self.determineScore(
                    a_to_b_transition, c_to_d_transition, semantic_figures['A'], semantic_figures['B'], semantic_figures['C'], curr)
                #print("a to b above, a to c below")
                score += self.determineScore(a_to_c_transition,
                                             b_to_d_transition, semantic_figures['A'], semantic_figures['B'], semantic_figures['C'], curr)
                scores[figure_id] = score

            highest = max(scores, key=scores.get)
            print(scores)
            return highest
        else:
            s = semantic_figures
            abc_trans = [self.determineTranslation(
                s['A'], s['B'], self.allg['A'], self.allg['B']), self.determineTranslation(s['B'], s['C'], self.allg['B'], self.allg['C'])]
            def_trans = [self.determineTranslation(
                s['D'], s['E'], self.allg['D'], self.allg['E']), self.determineTranslation(s['E'], s['F'], self.allg['E'], self.allg['F'])]
            adg_trans = [self.determineTranslation(
                s['A'], s['D'], self.allg['A'], self.allg['D']), self.determineTranslation(s['D'], s['G'], self.allg['D'], self.allg['G'])]
            beh_trans = [self.determineTranslation(
                s['B'], s['E'], self.allg['B'], self.allg['E']), self.determineTranslation(s['E'], s['H'], self.allg['E'], self.allg['H'])]
            ae_trans = self.determineTranslation(
                s['A'], s['E'], self.allg['A'], self.allg['E'])
            for figure_id in range(1, 9):
                self.i = figure_id
                # print(figure_id)
                curr = s[str(figure_id)]
                curr_gray = self.allg[str(figure_id)]
                ei_trans = self.determineTranslation(
                    s['E'], curr, self.allg['E'], curr_gray)
                cfi_trans = [self.determineTranslation(
                    s['C'], s['F'], self.allg['C'], self.allg['F']), self.determineTranslation(s['F'], curr, self.allg['F'], curr_gray)]
                ghi_trans = [self.determineTranslation(
                    s['G'], s['H'], self.allg['G'], self.allg['H']), self.determineTranslation(s['H'], curr, self.allg['H'], curr_gray)]
                score = self.determineScore(
                    ae_trans, ei_trans, s['A'], s['E'], s['E'], curr)
                score += self.determineScore3(
                    abc_trans, def_trans, ghi_trans, s['A'], s['B'], s['C'], s['D'], s['E'], s['F'], s['G'], s['H'], curr)
                score += self.determineScore3(
                    adg_trans, beh_trans, cfi_trans, s['A'], s['B'], s['C'], s['D'], s['E'], s['F'], s['G'], s['H'], curr)
                scores[figure_id] = score
            # print(scores)
            highest = max(scores, key=scores.get)
            stop = timeit.default_timer()
            #print(problem.name + " runtime: " + str(stop - start))
            print(scores)
            return highest

    def gather_pixel_ranges(self, semantic_figures):
        for key in semantic_figures:
            xvals = []
            yvals = []
            for di in semantic_figures[key]:
                xvals.append(di['xrange'][0])
                xvals.append(di['xrange'][1])
                yvals.append(di['yrange'][0])
                yvals.append(di['yrange'][1])
            if xvals == [] or yvals == []:
                self.isEmptySet = True
                return
            maxx = max(xvals)
            minx = min(xvals)
            maxy = max(yvals)
            miny = min(yvals)
            self.xrange[key] = (minx, maxx)
            self.yrange[key] = (miny, maxy)

    def find_shapes(self, figures):  # figures = dictionary containing all the images
        ret = {}
        for figure in figures:
            # /* BEGIN CODE FROM https://www.geeksforgeeks.org/how-to-detect-shapes-in-images-in-python-using-opencv/ */
            image = cv2.imread(figures[figure].visualFilename)
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, threshold = cv2.threshold(
                grayscale, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # contours = corners. HAIN_APPROX_NONE would give the list of all points instead of corner points
            i = 0
            #shape_id = 0
            figure_dict = {}
            #vs = []
            # for contour in contours:  # countour = 1 shapes corners. Note that a circle is all corners
            # if i == 0:
            #i = 1
            # continue
            # numCorners = cv2.approxPolyDP(
            # contour, 0.01 * cv2.arcLength(contour, True), True)
            # /* END CODE FROM https://www.geeksforgeeks.org/how-to-detect-shapes-in-images-in-python-using-opencv/ */
            # print(shape_id)
            # print(numCorners)
            # figure_dict[shape_id] = numCorners  # contour?
            #shape_id += 1
            #ret[figure] = figure_dict

            vs = []
            idx = 0
            shape_id = 0
            for contour in contours:
                if idx == 0:
                    # print(figure)
                    # print(contour)
                    # print(contour.shape)
                    if contour.shape[0] == 4:
                        vs.append(None)
                        idx = 1
                        continue
                    else:
                        contour = np.reshape(
                            contour, (contour.shape[0], contour.shape[2]))
                        idx_del = 0
                        del_li = []
                        for row in contour:
                            if (row[0] == 0 and row[1] == 0) or (row[0] == 0 and row[1] == 183) or (row[0] == 183 and row[1] == 183) or (row[0] == 183 and row[1] == 0):
                                del_li.append(idx_del)
                            idx_del += 1
                        # print(del_li)
                        contour = np.delete(contour, tuple(del_li), axis=0)
                        # print(contour)
                        # print("ssjss")
                numCorners = cv2.approxPolyDP(
                    contour, 0.01 * cv2.arcLength(contour, True), True)
                # print("vs")
                # print(vs)
                # print("numcorns")
                # print(numCorners)
                # print(numCorners.shape)
                if len(vs) > 1 and (len(numCorners) == len(vs[-1]) or (len(numCorners) > 8 and len(vs[-1] > 8))):
                    # print(figure)
                    # print(len(contours))
                    hollow = self.determineIsHollow(numCorners, vs[-1])
                    # print(hollow)
                    # print(figure_dict)
                    if hollow:
                        figure_dict[shape_id -
                                    1] = [figure_dict[shape_id - 1][0], "hollow"]
                        idx += 1
                        continue
                figure_dict[shape_id] = [numCorners, "nothollow"]
                shape_id += 1
                idx += 1
                vs.append(numCorners)
                # print(figure_dict)
            self.allg[figure] = self.modified_grayscale(grayscale)
            ret[figure] = figure_dict
        return ret

    def modified_grayscale(self, gray):
        g = np.where(gray > 0, 1, gray)
        # print(g)
        return g

    def determineIsHollow(self, corners1, corners2):
        # print(corners1)
        #print("corners 1 above 2 below")
        # print(corners2)
        corners2 = np.reshape(corners2, (corners2.shape[0], corners2.shape[2]))
        corners1 = np.reshape(corners1, (corners1.shape[0], corners1.shape[2]))
        correct = 0
        total = -1
        for corner in corners1:
            toBreak = False
            total += 1
            if correct != total:
                return False
            for i in range(corner[0] - 8, corner[0] + 11):
                if toBreak:
                    break
                for j in range(corner[1] - 8, corner[1] + 11):
                    if [i, j] in corners2.tolist():
                        toBreak = True
                        correct += 1
                        break
        return True

    def generate_networks(self, labeled):
        ret = {}
        for figure_id in labeled:
            shapes = []
            for shape_id in labeled[figure_id]:
                shape_dict = {}
                # labeled[figure_id][shape_id][1] holds if it is hollow or not
                reshaper = labeled[figure_id][shape_id][0].shape
                curr_shape = labeled[figure_id][shape_id][0].reshape(
                    reshaper[0], reshaper[2])
                xmax, ymax = curr_shape.max(axis=0)
                xmin, ymin = curr_shape.min(axis=0)
                shape_dict["id"] = shape_id
                shape_dict["xrange"] = (xmin, xmax)
                shape_dict["yrange"] = (ymin, ymax)
                shape_dict["numcorners"] = len(labeled[figure_id][shape_id][0])
                shape_dict["corners"] = curr_shape
                if labeled[figure_id][shape_id][1] == "hollow":
                    shape_dict["isHollow"] = True
                else:
                    shape_dict["isHollow"] = False
                # print("shapedytc")
                # print(shape_dict)
                if len(shapes) > 0:
                    for psd in shapes:  # psd = previous shape dict
                        if psd["xrange"][0] < shape_dict["xrange"][0] and psd["xrange"][1] > shape_dict["xrange"][1] and psd["yrange"][0] < shape_dict["yrange"][0] and psd["yrange"][1] > shape_dict["yrange"][1]:
                            shape_dict["inside"] = psd["id"]
                            continue
                        if (psd["xrange"][0] + psd["xrange"][1]) / 2 < (shape_dict["xrange"][0] + shape_dict["xrange"][1]) / 2:
                            shape_dict["rightof"] = psd["id"]
                        if (psd["xrange"][0] + psd["xrange"][1]) / 2 > (shape_dict["xrange"][0] + shape_dict["xrange"][1]) / 2:
                            shape_dict["leftof"] = psd["id"]
                        if (psd["yrange"][0] + psd["yrange"][1]) / 2 < (shape_dict["yrange"][0] + shape_dict["yrange"][1]) / 2:
                            shape_dict["above"] = psd["id"]
                        if (psd["yrange"][0] + psd["yrange"][1]) / 2 > (shape_dict["yrange"][0] + shape_dict["yrange"][1]) / 2:
                            shape_dict["below"] = psd["id"]
                shapes.append(shape_dict)
            ret[figure_id] = shapes
        return ret

    def determineTranslation(self, semA, semB, grayA, grayB):
        trans_dict = {}
        if len(semA) == len(semB):
            for i in range(len(semA)):
                acorners = semA[i]["corners"]
                bcorners = semB[i]["corners"]
                if self.isUnchanged(acorners, bcorners):
                    trans_dict[i] = "unchanged"
                elif self.isReflectedud(acorners, bcorners):
                    trans_dict[i] = "reflected ud"
                elif self.isReflectedlr(acorners, bcorners):
                    trans_dict[i] = "reflected lr"
                else:
                    trans_dict[i] = "undecided"  # can add scaling, rotations
        else:
            for i in range(max(len(semA), len(semB))):
                trans_dict[i] = "diff"
        numAcorners = 0
        numBcorners = 0
        for i in range(len(semA)):
            numAcorners += semA[i]["numcorners"]
        for i in range(len(semB)):
            numBcorners += semB[i]["numcorners"]
        trans_dict["cornersadded"] = numBcorners - numAcorners
        trans_dict["shapesadded"] = len(semB) - len(semA)
        trans_dict["mse"] = self.mse(grayA, grayB)
        return trans_dict

    def mse(self, a, b):
        h, w = a.shape
        diff = cv2.subtract(a, b)
        error = np.sum(diff**2)
        mse = error / (float(h * w))
        if mse > 1:
            print("what")
        #print(1 - mse)
        return 1 - mse

    def isUnchanged(self, acorners, bcorners):
        count = 0
        if len(acorners) != len(bcorners):
            return False
        for acorner in acorners:
            toBreak = False
            for x in range(acorner[0] - 1, acorner[0] + 2):
                if toBreak:
                    break
                for y in range(acorner[1] - 1, acorner[1] + 2):
                    for bcorner in bcorners:
                        if [x, y] == [bcorner[0], bcorner[1]]:
                            count += 1
                            toBreak = True
                            break
        return count == len(bcorners)

    def isReflectedud(self, acorners, bcorners):
        isCircle = False
        if len(acorners) > 8:
            isCircle = True
        if len(acorners) != len(bcorners) and not isCircle:
            return False
        correct = 0
        reflectb = []
        for corner in bcorners:
            reflect_corner = [183 - corner[0], corner[1]]
            reflectb.append(reflect_corner)
        for acorner in acorners:
            toBreak = False
            for x in range(acorner[0] - 2, acorner[0] + 3):
                if toBreak:
                    break
                for y in range(acorner[1] - 2, acorner[1] + 3):
                    for b in reflectb:
                        if [x, y] == [b[0], b[1]]:
                            correct += 1
                            toBreak = True
                            break
        total = max(len(acorners), len(bcorners))
        if isCircle:
            if correct + 1 == total or correct - 1 == total or correct + 2 == total or correct - 2 == total:
                return True
            else:
                return False
        return correct == total

    def isReflectedlr(self, acorners, bcorners):
        isCircle = False
        if len(acorners) > 8:
            isCircle = True
        if len(acorners) != len(bcorners) and not isCircle:
            return False
        correct = 0
        reflectb = []
        for corner in bcorners:
            reflect_corner = [corner[0], 183 - corner[1]]
            reflectb.append(reflect_corner)
        for acorner in acorners:
            toBreak = False
            for x in range(acorner[0] - 2, acorner[0] + 3):
                if toBreak:
                    break
                for y in range(acorner[1] - 2, acorner[1] + 3):
                    for b in reflectb:
                        if [x, y] == [b[0], b[1]]:
                            correct += 1
                            toBreak = True
                            break
        total = max(len(acorners), len(bcorners))
        if isCircle:
            if correct + 1 == total or correct - 1 == total or correct + 2 == total or correct - 2 == total:
                return True
            else:
                return False
        return correct == total

    def determineScore(self, ab, cd, semA, semB, semC, semD):
        score = 0
        if len(ab) != len(cd):
            pass
        else:
            for shapeid in ab:
                if ab[shapeid] == cd[shapeid]:
                    if ab[shapeid] == "unchanged":
                        score += 10
                    elif ab[shapeid] == "reflected ud" or ab[shapeid] == "reflected lr":
                        score += 5
        if ab["cornersadded"] == cd["cornersadded"]:
            score += 4
        if ab["shapesadded"] == cd["shapesadded"]:
            score += 4
        diff = abs(ab['mse'] - cd['mse'])
        score += (1 - diff) * 100
        return score

    def determineScore3(self, abct, deft, ghit, sA, sB, sC, sD, sE, sF, sG, sH, sI):
        # print(sA)
        # print(abct)
        #print("abc above, deft beliow")
        # print(deft)
        # print("abc")
        # print(abct)
        # print("ghi")
        # print(ghit)
        score = 0
        if [len(abct[0]), len(abct[1])] == [len(ghit[0]), len(ghit[1])]:
            # print("heeh")
            for i in range(0, 2):
                for shape_id in abct[i]:
                    # print("uuuuu")
                    # print(abct[i])
                    if abct[i][shape_id] == ghit[i][shape_id]:
                        if abct[i][shape_id] == "unchanged":
                            #print("found and cufnnf")
                            score += 8
                        elif abct[i][shape_id] == "reflected ud" or abct[i][shape_id] == "reflected lr":
                            score += 6
                        else:
                            score += 2
        if [abct[0]["cornersadded"], abct[1]["cornersadded"]] == [ghit[0]["cornersadded"], ghit[1]["cornersadded"]]:
            score += 3
        if [abct[0]["shapesadded"], abct[1]["shapesadded"]] == [ghit[0]["shapesadded"], ghit[1]["shapesadded"]]:
            score += 3

        if [len(deft[0]), len(deft[1])] == [len(ghit[0]), len(ghit[1])]:
            for i in range(0, 2):
                for shape_id in deft[i]:
                    if deft[i][shape_id] == ghit[i][shape_id]:
                        if deft[i][shape_id] == "unchanged":
                            score += 8
                        elif deft[i][shape_id] == "reflected ud" or deft[i][shape_id] == "reflected lr":
                            score += 6
                        else:
                            score += 2
        if [deft[0]["cornersadded"], deft[1]["cornersadded"]] == [ghit[0]["cornersadded"], ghit[1]["cornersadded"]]:
            score += 3
        if [deft[0]["shapesadded"], deft[1]["shapesadded"]] == [ghit[0]["shapesadded"], ghit[1]["shapesadded"]]:
            score += 3
        for shape_id in ghit[0]:
            try:
                if ghit[0][shape_id] == ghit[1][shape_id]:
                    score += 1
            except:
                break
        #diff = abs(abct[1]['mse'] - deft[1]['mse'])
        # diff = (abs(abct[0]['mse'] - ghit[0]['mse']),
                # abs(abct[1]['mse'] - ghit[1]['mse']))
        #diff = (diff[0]**2 + diff[1]**2) ** .5
        # print("score")
        # print(score)
        # print(diff)
        # print(score)
        # print(self.i)
        abc_ratio = abct[0]['mse'] / abct[1]['mse']
        def_ratio = deft[0]['mse'] / deft[1]['mse']
        ghi_ratio = ghit[0]['mse'] / ghit[1]['mse']
        diff = abs(abc_ratio - ghi_ratio)
        #print("mse:  " + str(diff))
        score += (1 - diff) * 1000
        # diff2 = (abs(deft[0]['mse'] - ghit[0]['mse']),
        # abs(deft[1]['mse'] - ghit[1]['mse']))
        # print(score)
        diff2 = abs(def_ratio - ghi_ratio)
        #print("mse:  " + str(diff2))
        #diff2 = (diff2[0]**2 + diff2[1]**2) ** .5
        score += (1 - diff2) * 1000
        # print(score)
        if self.isEmptySet:
            return score
        ab_range_dif_min = abs(self.xrange['A'][0] - self.xrange['B'][0])
        bc_range_dif_min = abs(self.xrange['B'][0] - self.xrange['C'][0])
        de_range_dif_min = abs(self.xrange['D'][0] - self.xrange['E'][0])
        ef_range_dif_min = abs(self.xrange['E'][0] - self.xrange['F'][0])
        gh_range_dif_min = abs(self.xrange['G'][0] - self.xrange['H'][0])
        hi_range_dif_min = abs(
            self.xrange['H'][0] - self.xrange[str(self.i)][0])
        if bc_range_dif_min == 0:
            abc_xrangemin_rat = 0
        else:
            abc_xrangemin_rat = ab_range_dif_min / bc_range_dif_min
        if ef_range_dif_min == 0:
            def_xrangemin_rat = 0
        else:
            def_xrangemin_rat = de_range_dif_min / ef_range_dif_min
        if hi_range_dif_min == 0:
            ghi_xrangemin_rat = 0
        else:
            ghi_xrangemin_rat = gh_range_dif_min / hi_range_dif_min
        diff3 = abs(abc_xrangemin_rat - ghi_xrangemin_rat)
        if diff3 < 10:
            score += 50000
        diff4 = abs(def_xrangemin_rat - ghi_xrangemin_rat)
        if diff4 < 10:
            score += 50000

        ab_xrange_dif_max = abs(self.xrange['A'][1] - self.xrange['B'][1])
        bc_xrange_dif_max = abs(self.xrange['B'][1] - self.xrange['C'][1])
        de_xrange_dif_max = abs(self.xrange['D'][1] - self.xrange['E'][1])
        ef_xrange_dif_max = abs(self.xrange['E'][1] - self.xrange['F'][1])
        gh_xrange_dif_max = abs(self.xrange['G'][1] - self.xrange['H'][1])
        hi_xrange_dif_max = abs(
            self.xrange['H'][1] - self.xrange[str(self.i)][1])
        if bc_xrange_dif_max == 0:
            abc_xrangemax_rat = 0
        else:
            abc_xrangemax_rat = ab_xrange_dif_max / bc_xrange_dif_max
        if ef_xrange_dif_max == 0:
            def_xrangemax_rat = 0
        else:
            def_xrangemax_rat = de_xrange_dif_max / ef_xrange_dif_max
        if hi_xrange_dif_max == 0:
            ghi_xrangemax_rat = 0
        else:
            ghi_xrangemax_rat = gh_xrange_dif_max / hi_xrange_dif_max
        diff5 = abs(abc_xrangemax_rat - ghi_xrangemax_rat)
        if diff5 < 10:
            score += 50000
        diff6 = abs(def_xrangemax_rat - ghi_xrangemax_rat)
        if diff6 < 10:
            score += 50000

        ab_yrange_dif_max = abs(self.yrange['A'][1] - self.yrange['B'][1])
        bc_yrange_dif_max = abs(self.yrange['B'][1] - self.yrange['C'][1])
        de_yrange_dif_max = abs(self.yrange['D'][1] - self.yrange['E'][1])
        ef_yrange_dif_max = abs(self.yrange['E'][1] - self.yrange['F'][1])
        gh_yrange_dif_max = abs(self.yrange['G'][1] - self.yrange['H'][1])
        hi_yrange_dif_max = abs(
            self.yrange['H'][1] - self.yrange[str(self.i)][1])
        if bc_yrange_dif_max == 0:
            abc_yrangemax_rat = 0
        else:
            abc_yrangemax_rat = ab_yrange_dif_max / bc_yrange_dif_max
        if ef_yrange_dif_max == 0:
            def_yrangemax_rat = 0
        else:
            def_yrangemax_rat = de_yrange_dif_max / ef_yrange_dif_max
        if hi_yrange_dif_max == 0:
            ghi_yrangemax_rat = 0
        else:
            ghi_yrangemax_rat = gh_yrange_dif_max / hi_yrange_dif_max
        diff7 = abs(abc_yrangemax_rat - ghi_yrangemax_rat)
        if diff7 < 10:
            score += 50000
        diff8 = abs(def_yrangemax_rat - ghi_yrangemax_rat)
        if diff8 < 10:
            score += 50000

        ab_yrange_dif_min = abs(self.yrange['A'][0] - self.yrange['B'][0])
        bc_yrange_dif_min = abs(self.yrange['B'][0] - self.yrange['C'][0])
        de_yrange_dif_min = abs(self.yrange['D'][0] - self.yrange['E'][0])
        ef_yrange_dif_min = abs(self.yrange['E'][0] - self.yrange['F'][0])
        gh_yrange_dif_min = abs(self.yrange['G'][0] - self.yrange['H'][0])
        hi_yrange_dif_min = abs(
            self.yrange['H'][0] - self.yrange[str(self.i)][0])
        if bc_yrange_dif_min == 0:
            abc_yrangemin_rat = 0
        else:
            abc_yrangemin_rat = ab_yrange_dif_min / bc_yrange_dif_min
        if ef_yrange_dif_min == 0:
            def_yrangemin_rat = 0
        else:
            def_yrangemin_rat = de_yrange_dif_min / ef_yrange_dif_min
        if hi_yrange_dif_min == 0:
            ghi_yrangemin_rat = 0
        else:
            ghi_yrangemin_rat = gh_yrange_dif_min / hi_yrange_dif_min
        diff9 = abs(abc_yrangemin_rat - ghi_yrangemin_rat)
        if diff9 < 10:
            score += 50000
        diff10 = abs(def_yrangemin_rat - ghi_yrangemin_rat)
        if diff10 < 10:
            score += 50000
        return score
