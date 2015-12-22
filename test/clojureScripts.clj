(ns my-example-clj.ok)

(defn sum
  ([values]
   (sum values 0))
  ([values result]
   (if (empty? values)
     result
     (recur (rest values) (+ (first values) result)))))


(sum [12 3 56])


(defn check-type [values]
  (cond
    (= (inc (count values)) (count (conj values {:test 1} {:test 2}))) :map
    (= (inc (count values)) (count (conj values {:test 1} {:test 1}))) :set
    (= :test (last (conj values :dummy :test))) :vector
    :else :list))

(map check-type [{:test1 1} #{:test1 1} [:test1 1] (:test1 1)])

;http://textfiles.com/stories/gulliver.txt
(def common-english-words
  (-> (slurp "http://www.textfixer.com/resources/common-english-words-with-contractions.txt")
      (clojure.string/split #",")
      set))

(def input
  (slurp "http://textfiles.com/stories/gulliver.txt"))

;;thread macro last
(->> input
     (re-seq #"[\w|']+")
     (map #(clojure.string/lower-case %))
     (remove common-english-words)
     frequencies
     (sort-by val)
     reverse)
