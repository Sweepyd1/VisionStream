import numpy as np

def perpendicular_dist(point, start, end):
    line = end - start
    if np.allclose(line, 0): # если конец отрезка совпадает с его началом (start ~= end)
        return np.linalg.norm(point-start) # то берём перпендикуляр - как расстояние от старта до точки 
    return abs(np.cross(line, point-start)) / np.linalg.norm(line) # np.cross - ворзвращает векторное произведениие. ∣a×b∣ == ∣a∣∣b∣sin(θ) == S трапеции. чтобы найти l перпендикуляра -> делим S паралеллаграма на длину стороны, куда опускается перпендикуляр

def approxPolyDP(points, epsilon, closed=True):

    """
    Алгоритм Ramer–Douglas–Peucker.

    points: numpy-массив формы (N, 2) — точки контура по порядку.
    epsilon: допуск аппроксимации в пикселях (макс. разрешённое отклонение).
    closed: замкнут ли контур (соединять ли конец с началом).
    """

    if points is None or len(points) == 0: # если пустой массив точек -> выдаём пустой результат
        return np.empty((0,2), dtype=points.dtype if isinstance(points, np.ndarray) else np.float32)
    
    pts = (np.asarray(points, dtype = np.float32))
    if closed: # если контур замкнут, то дбавляем начльную точку в конец массива, чтобы сшить начало и конец, в случае необходимости взятия отрезков 
        pts = np.vstack([pts, pts[0]]) # добавляем в конец массива точку начала контура

    keep = np.zeros(len(pts), dtype=bool) # строим массив буллевых точек(t - берём в итоговый контур, f - не берём) размерности имеющихся точек на контурах 
    keep[0] = True
    keep[-1] = False

    stack = [(0, len(pts) - 1)] # стек отрезков (индексы начала и конца отрезка)    
    
    while stack:

        s, e = stack.pop() 
        if e<=s+1:
            continue # если отрезок слишком мал, то пропускаем его "s"|----------------|"e"

        """ 
        Ищем точку с максимальным отклонением от прямой 

        """

        max_dist = 0
        index_max = -1
        
        for i in range(s+1, e): # перебираем все точки между концами отрезка
            d = perpendicular_dist(pts[i], pts[s], pts[e])
            if d > max_dist:
                max_dist = d
                index_max = i
        
        if max_dist > float(epsilon):
            keep[index_max] = True # если отклонение больше допустимого, то сохраняем точку в итоговый контур
            stack.append((s, index_max))
            stack.append((index_max, e))
        
    approx = pts[keep] # итоговый контур - точки, которые нужно сохранить

    if closed:
        if not np.array_equal(approx[0], approx[-1]): # если контур замкнут, то нужно проверить, что первая и последняя точка совпадают
            approx = np.vstack([approx, approx[0]]) # если не совпадают, то добавляем в конец контура начальную точку
        
    return approx.astype(points.dtype, copy=False)
    
    



