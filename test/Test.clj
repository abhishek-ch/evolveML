(ns Test)
(def headerLinks
  [{:name "Link1" :href "http://www.google.com" :class "glyphicon glyphicon-chevron-right"}
   {:name "Link2" :href "http://www.google.com" :class "glyphicon glyphicon-user"}
   {:name "Link3" :href "http://www.google.com" :class "glyphicon glyphicon-lock"}
   {:name "Link4" :href "http://www.google.com" :class "glyphicon glyphicon-cog"}]
  )



(defn newset []
  (doseq [record headerLinks
          [k v] record]
    (fn [k v]
      [:div
       [:val {:key k :value v}]
       ]
      ))
  )

(println newset)

(doseq [record headerLinks
        [k v] record]
  (

    ; println (str k " " v))

    )
  )


(defn dollme [n]
  (println "DOLL "n)
  )

(defn callme [n x]
  (println n)
  (dollme x)
  )



(print (map #(callme (:class %1)(:href %1) ) headerLinks))

;(print (map #(println (:class %1)) headerLinks))

  (def config [{:host "test", :port 1}, {:host "testtest", :port 2}])
  (map #(print (:host %1)) config)


