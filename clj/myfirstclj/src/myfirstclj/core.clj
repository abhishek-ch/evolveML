(ns myfirstclj.core
  (:use compojure.core
        ring.middleware.json
        ring.util.response)
  (:require [compojure.route :as route]
            [myfirstclj.view :as view])
  )

;;https://www.youtube.com/watch?v=jOX0uK3jsbI&index=20&list=WL
(defn foo
  "I don't do a whole lot."
  [x]
  (str "Hello, "x))

(defroutes my_routes
  (GET "/" [] (view/callme) )
  (GET "/rest" [] (response {:email "abhishek.create@gmail.com"}))
  (route/resources "/")
  )


(def app
  (wrap-json-response my_routes))