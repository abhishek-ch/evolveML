(ns myfirstclj.controller.contentctlr
  (:require [compojure.core :refer [defroutes GET POST]]
            [clojure.string :as str]
            [ring.util.response :as ring]
            [myfirstclj.views.contents :as view]
            )
  )


(defn index []
  (view/index))
(defn not-found []
  (view/not-found))


(defroutes routes
           (GET  "/" [] (index))
           (POST "/" [] (not-found)))
