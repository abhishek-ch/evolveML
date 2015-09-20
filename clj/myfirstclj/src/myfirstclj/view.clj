(ns
  ^{:author abc}
  myfirstclj.view
  (:use  hiccup.page hiccup.element))


(defn callme
  []
  (html5
    [:html
     [:title "Bond is Back"]
     [:body "Abhishek Testing"]
      ]
    )
  )