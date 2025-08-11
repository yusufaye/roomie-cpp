// message.h
#ifndef MESSAGE_H
#define MESSAGE_H

#include <map>
#include <string>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

enum class Type
{
    QUERY,
    HELLO,
    FINISHED,
    REGISTER,
    PROFILE_DATA,
    DEPLOYED,
    STOP,
    DEPLOY,
};

struct MessageTypeInfo
{
    std::string name;
    int value;
};

std::map<Type, std::string> type2string = {
    {Type::QUERY, "QUERY"},
    {Type::HELLO, "HELLO"},
    {Type::FINISHED, "FINISHED"},
    {Type::REGISTER, "REGISTER"},
    {Type::PROFILE_DATA, "PROFILE_DATA"},
    {Type::DEPLOYED, "DEPLOYED"},
    {Type::STOP, "STOP"},
    {Type::DEPLOY, "DEPLOY"},
};

class Message
{
public:
    Message() {}

    Message(const std::string &type)
        : type_(type) {}

    Message(const std::string &type, const json &data)
        : type_(type), data_(data)
    {
    }

    Message(const double timestamp, const std::string &type, const json &data)
        : timestamp_(timestamp), type_(type), data_(data) {}

    double get_timestamp() const { return timestamp_; }
    std::string get_type() const { return type_; }
    json get_data() const { return data_; }

    void append_data(std::string key, std::string value)
    {
        data_[key] = value;
    }

    std::string serialize() const
    {
        // convert to JSON: copy each value into the JSON object
        json j = {{"timestamp", timestamp_}, {"type", type_}, {"data", data_}};
        return j.dump();
    }
    
    void deserialize(const std::string &s)
    {
        auto j = json::parse(s);
        
        // convert from JSON: copy each value from the JSON object
        data_       = j["data"];
        type_       = j["type"].get<std::string>();
        timestamp_  = j["timestamp"].get<double>();
    }

    std::string to_string() const
    {
        return "Message('timestamp': " + std::to_string(timestamp_) +
               ", 'type': " + type_ + ")";
    }

private:
    double timestamp_ = 0.0;
    std::string type_;
    json data_;
};

#endif // MESSAGE_H